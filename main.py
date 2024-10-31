import asyncio
import numpy as _
import pygame as _
from modules.game import *


async def main():
    backend_url = "http://localhost:81"
    game = DefaultMarsGame(backend_url)
    # game = TestEarthGeostationary()
    await game.rungame()


asyncio.run(main())
