from logging import INFO, Formatter, Logger, StreamHandler, getLogger

import boto3
import typer
import xarray as xr
from dask.distributed import Client
from dep_tools.aws import object_exists
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.grids import PACIFIC_GRID_10
from dep_tools.loaders import OdcLoader
from dep_tools.namers import S3ItemPath
from dep_tools.processors import Processor
from dep_tools.searchers import PystacSearcher
from dep_tools.stac_utils import StacCreator
from dep_tools.task import AwsStacTask as Task
from dep_tools.writers import AwsDsCogWriter
from odc.geo import Geometry
from odc.stac import configure_s3_access
from typing_extensions import Annotated
from xarray import DataArray

# NIU uv run src/run.py --tile-id 77,19 --year 2024 --version 0.3.0
# NRU uv run src/run.py --tile-id 50,41 --year 2024 --version 0.3.0
# FJI_Coral_Coast uv run src/run.py --tile-id 84,63 --year 2024 --version 0.3.0

OUTPUT_NODATA = 255

def get_logger(region_code: str, name: str) -> Logger:
    """Set up a simple logger"""
    console = StreamHandler()
    time_format = "%Y-%m-%d %H:%M:%S"
    console.setFormatter(
        Formatter(
            fmt=f"%(asctime)s %(levelname)s ({region_code}):  %(message)s",
            datefmt=time_format,
        )
    )
    log = getLogger(name)
    log.addHandler(console)
    log.setLevel(INFO)
    return log


class MangrovesProcessor(Processor):
    def __init__(self, areas: Geometry):
        super().__init__()
        self.areas = areas

    def process(self, data: DataArray) -> DataArray:
        data = data.squeeze()

        # AMMI
        nir = data["nir"]
        swir = data["swir16"]
        red = data["red"]
        green = data["green"]
        blue = data["blue"]

        ammi = ((nir - red) / (red + swir)) * ((nir - swir) / (swir - 0.65 * red))
        ammi = ammi.to_dataset(name="ammi")

        # AMMI_THRESHOLD = 4.0 - 20
        AMMI_THRESHOLD = range(4, 20)
        mangrove_mask = ammi.ammi >= list(AMMI_THRESHOLD)[0]

        num_vals = len(AMMI_THRESHOLD)
        for i, val in enumerate(AMMI_THRESHOLD, 1):
            density_percentage = 10 + (i - 1) * (90 / (num_vals - 1))
            mangrove_mask = xr.where(
                ammi.ammi >= val, density_percentage, mangrove_mask
            )

        # Convert boolean mask to uint8 (0/1) and attach geospatial metadata for saving
        mangrove_mask = mangrove_mask.astype("uint8")
        mangrove_mask = mangrove_mask.compute()

        # remove null
        data["mangroves"] = mangrove_mask.where(mangrove_mask != 0, drop=True)

        # Morphological Filters and Elevation Masking
        data["ndwi"] = (data.green - data.nir) / (data.green + data.nir)
        data["mndwi"] = (data.green - data.swir16) / (data.green + data.swir16)

        # water mask
        water = (data.mndwi + data.ndwi).squeeze() < 0
        water_mask = mask_cleanup(water, [["dilation", 5], ["erosion", 5]])
        data["mangroves"] = util.apply_mask(data["mangroves"], water_mask)

        # elevation mask (40-50m)
        data["mangroves"] = util.mask_elevation(data["mangroves"], threshold=50)

        # Only keep the mangroves band and set nodata
        data = data[["mangroves"]].astype("uint8")
        data.mangroves.odc.nodata = OUTPUT_NODATA

        return data


def main(
    tile_id: Annotated[str, typer.Option()],
    year: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    output_bucket: str = "dep-public-staging",
    memory_limit: str = "50GB",
    dataset_id: str = "mangroves",
    n_workers: int = 4,
    threads_per_worker: int = 32,
    decimated: bool = False,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    log = get_logger(tile_id, "MANGROVES")
    log.info("Starting processing...")

    grid = PACIFIC_GRID_10

    tile_index = tuple(int(i) for i in tile_id.split(","))
    geobox = grid.tile_geobox(tile_index)

    if decimated:
        log.warning("Running at 1/10th resolution")
        geobox = geobox.zoom_out(10)

    # Make sure we can access S3
    log.info("Configuring S3 access")
    configure_s3_access(cloud_defaults=True)

    client = boto3.client("s3")

    itempath = S3ItemPath(
        bucket=output_bucket,
        sensor="s2",
        dataset_id=dataset_id,
        version=version,
        time=year,
    )
    stac_document = itempath.stac_path(tile_id)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite and object_exists(output_bucket, stac_document, client=client):
        log.info(f"Item already exists at {stac_document}")
        # This is an exit with success
        raise typer.Exit()

    catalog = "https://stac.digitalearthpacific.org"
    collection = "dep_s2_geomad"

    searcher = PystacSearcher(catalog=catalog, collections=[collection], datetime=year)

    loader = OdcLoader(
        bands=[
            "nir",
            "red",
            "blue",
            "green",
            "green",
            "swir16",
        ],
        # chunks=[-1, 2048, 2048],
        groupby="solar_day",
        fail_on_error=False,
        clip_to_area=False,
        chunks={"x": 2048, "y": 2048},
    )

    processor = MangrovesProcessor(areas=geobox)

    # Custom writer so we write multithreaded
    writer = AwsDsCogWriter(itempath, write_multithreaded=True)

    # STAC making thing
    stac_creator = StacCreator(
        itempath=itempath, remote=True, make_hrefs_https=True, with_raster=True
    )

    try:
        with Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
        ):
            log.info(
                (
                    f"Started dask client with {n_workers} workers "
                    f"and {threads_per_worker} threads with "
                    f"{memory_limit} memory"
                )
            )
            paths = Task(
                itempath=itempath,
                id=tile_index,
                area=geobox,
                searcher=searcher,
                loader=loader,
                processor=processor,
                writer=writer,
                logger=log,
                stac_creator=stac_creator,
            ).run()
    except EmptyCollectionError:
        log.info("No items found for this tile")
        raise typer.Exit()  # Exit with success
    except Exception as e:
        log.exception(f"Failed to process with error: {e}")
        raise typer.Exit(code=1)

    log.info(
        f"Completed processing. Wrote {len(paths)} items to https://{output_bucket}.s3.us-west-2.amazonaws.com/{ stac_document}"
    )


if __name__ == "__main__":
    typer.run(main)
