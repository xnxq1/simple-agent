import click
import uvicorn

from app.main import create_app


@click.group()
def cli():
    pass


@cli.command(short_help="Start api")
def start_api():
    uvicorn.run(create_app(), host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    cli()
