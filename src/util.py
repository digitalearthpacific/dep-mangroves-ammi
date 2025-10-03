#import cv2
#import numpy as np
from xarray import DataArray, Dataset
from odc.stac import load
from pystac_client import Client

def apply_mask(
    ds: Dataset,
    mask: DataArray,
    ds_to_mask: Dataset | None = None,
    return_mask: bool = False,
) -> Dataset:
    """Applies a mask to a dataset"""
    to_mask = ds if ds_to_mask is None else ds_to_mask
    masked = to_mask.where(mask)

    if return_mask:
        return masked, mask
    else:
        return masked


def mask_elevation(
    ds: Dataset,
    ds_to_mask: Dataset | None = None,
    threshold: float = 10,
    return_mask: bool = False,
) -> Dataset:
    """
    Mask elevation. Returns 1 for high areas, 0 for low
    """
    e84_catalog = "https://earth-search.aws.element84.com/v1/"
    e84_client = Client.open(e84_catalog)
    collection = "cop-dem-glo-30"

    items = e84_client.search(
        collections=[collection], bbox=list(ds.odc.geobox.geographic_extent.boundingbox)
    ).item_collection()

    # Using geobox means it will load the elevation data the same shape as the other data
    elevation = load(items, measurements=["data"], geobox=ds.odc.geobox).squeeze()

    # True where data is above elevation
    mask = elevation.data > threshold

    return apply_mask(ds, mask, ds_to_mask, return_mask)

"""
def filter_mask(closing_kernel_size, opening_kernel_size, mask):
    
    ## Closing filter: Remove empty pixels within mask
    # Create a kernel element which is closing_kernel_size^2 in size
    closing_kernel_element = (closing_kernel_size, closing_kernel_size)
    # Create a closing filter kernel
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               closing_kernel_element)
    # Apply closing filter to input mask
    mask_closed = cv2.morphologyEx(np.nan_to_num(mask), cv2.MORPH_CLOSE,
                                   closing_kernel)

    ## Opening filter: Removing filled pixels outside of mask
    # Create a kernel element which is closing_kernel_size^2 in size
    opening_kernel_element = (opening_kernel_size, opening_kernel_size)
    # Create an opening filter kernel
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               opening_kernel_element)
    # Apply opening filter to closed mask
    mask_closed_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN,
                                          opening_kernel)

    # Ensure the clipped areas remain clipped
    mask_closed_opened[mask_closed_opened == 0] = np.nan

    return mask_closed_opened
    
"""