
.ONESHELL:

default: lab

lab:
	uv sync
	BROWSER=firefox uv run --with jupyter jupyter-lab

upgrade:
	uv pip list --outdated
	rm uv.lock
	uv sync -U 
	uv lock --upgrade
	

env:
	. .venv/bin/activate

docker:
	docker build -t mangroves .
	
