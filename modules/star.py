import numpy as np
import pygame as pg

from modules.types import Coordinate


class Star:
    """
    Class that contains the coordinates of a Star
    """

    def __init__(self, pos: Coordinate) -> None:
        """
        initializes a Star

        Parameters:
            pos (Coordinate): The position of the Star on the surface in pixels
        """
        self.pos = pos

    def change_pos(
        self,
        dt: np.float64,
        vel: np.ndarray[np.float64, np.dtype[np.float64]],
        screensize: Coordinate,
    ) -> None:
        """
        A function to change the position of a star after a tick

        Parameters:
            dt (np.float64): The time since the last tick
            vel (np.ndarray[np.float64, np.dtype[np.float64]]): The velocity of the probe
            screensize (Coordinate): The size of the Screen
        """
        x, y = vel
        self.pos = (
            (self.pos[0] - dt * x * 0.5) % screensize[0],
            (self.pos[1] + dt * y * 0.5) % screensize[1],
        )

    @staticmethod
    def generate_stars(n: int, screen_size: Coordinate) -> list["Star"]:
        """
        Generates a list of stars on random positions on the screen

        Parameters:
            n (int): The number of Stars to generate
            screen_size (Coordinate): The size of the screen

        Returns:

        """
        return [
            Star(
                (
                    np.random.randint(0, screen_size[0]),
                    np.random.randint(0, screen_size[1]),
                )
            )
            for _ in range(n)
        ]

    @staticmethod
    def draw_stars(
        screen: pg.surface.Surface,
        stars: list["Star"],
        vel: np.ndarray[np.float64, np.dtype[np.float64]],
        dt: np.float64,
        screensize: Coordinate,
    ) -> None:
        """
        Draws the stars on the screen

        Parameters:
            screen (pg.surface.Surface): The screen to draw on
            stars (list["Star"]): The stars to draw
            vel (np.ndarray[np.float64, np.dtype[np.float64]]): The velocity of the probe
            dt (np.float64): The time since the last tick
            screensize (Coordinate): The size of the screen
        """
        for star in stars:
            pg.draw.circle(screen, "white", star.pos, 2)
            star.change_pos(dt, vel, screensize)
