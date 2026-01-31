import click
import uvicorn

from app.di import container
from app.main import AppBuilder


@click.group()
def cli():
    pass


@cli.command(short_help="Start api")
def start_api():
    app_builder = container.get(AppBuilder)
    uvicorn.run(app_builder.create_app(), host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    cli()
