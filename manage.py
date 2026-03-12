import logging

import click
import uvicorn

from app.di import container
from app.main import AppBuilder


@click.group()
def cli():
    pass


@cli.command(short_help="Start api")
def start_api():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app_builder = container.get(AppBuilder)
    uvicorn.run(app_builder.create_app(), host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    cli()
