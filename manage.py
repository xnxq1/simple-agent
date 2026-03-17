import asyncio
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

    async def run():
        async with container:
            app_builder = await container.get(AppBuilder)
            config = uvicorn.Config(app_builder.create_app(), host="0.0.0.0", port=8000, reload=False)
            server = uvicorn.Server(config)
            await server.serve()

    asyncio.run(run())


if __name__ == "__main__":
    cli()
