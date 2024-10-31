import numpy as np
import pygame as pg
import modules.constants as constants


class Body:
    """
    A class to represent a celestial body.

    Attributes:
        mass (np.float64): The mass of the body in kg
        radius (np.float64): The radius of the body in m
        color (pg.color.Color): The color of the body
        densitiy_at_surface (np.float64): The density of the atmosphere at the surface of the body in hPa
        temperature_gradient (np.float64): The temperature gradient of the atmosphere in K / m
        temperature_at_surface (np.float64): The temperature of the atmosphere at the surface of the body in K
        middle_atmospheric_molar_mass (np.float64): The molar mass of the atmosphere in kg / mol
    """

    def __init__(
        self,
        mass: np.float64,
        radius: np.float64,
        color: pg.color.Color,
        densitiy_at_surface: np.float64,
        temperature_gradient: np.float64,
        temperature_at_surface: np.float64,
        middle_atmospheric_molar_mass: np.float64,
    ) -> None:
        """
        Initialize the body with the given parameters.

        Parameters:
            mass (np.float64): The mass of the body in kg
            radius (np.float64): The radius of the body in m
            color (pg.color.Color): The color of the body
            densitiy_at_surface (np.float64): The density of the atmosphere at the surface of the body in hPa
            temperature_gradient (np.float64): The temperature gradient of the atmosphere in K / m
            temperature_at_surface (np.float64): The temperature of the atmosphere at the surface of the body in K
            middle_atmospheric_molar_mass (np.float64): The molar mass of the atmosphere in kg / mol
        """
        self.mass = mass
        self.radius = radius
        self.color = color
        self.densitiy_at_surface = densitiy_at_surface
        self.temperature_gradient = temperature_gradient
        self.temperature_at_surface = temperature_at_surface
        self.middle_atmospheric_molar_mass = middle_atmospheric_molar_mass

    def get_atmosperic_density(self, altitude: np.float64) -> np.float64:
        """
        Calculate the density of the atmosphere at the given altitude.
        uses the barometric formula (1)

        Parameters:
            altitude (np.float64): The altitude at which the density should be calculated in m

        Returns:
            np.float64: The density of the atmosphere at the given altitude in kg / m^3
        """
        G = constants.GRAVITATIONAL_CONSTANT
        R = constants.GAS_CONSTANT
        g = G * self.mass / self.radius**2  # (2)
        exp = (self.middle_atmospheric_molar_mass * g) / (R * self.temperature_gradient)

        inner = 1 - self.temperature_gradient * altitude / self.temperature_at_surface

        if inner < 0:
            return np.float64(0)

        return self.densitiy_at_surface * 100 * (inner**exp)


def test() -> None:
    """
    Run tests for the Body class.
    """
    body = Body(
        np.float64(6.39e23),
        np.float64(3389500),
        pg.color.Color("orange"),
        np.float64(7),
        np.float64(6.5e-3),
        np.float64(240.15),
        np.float64(42.8172e-3),
    )

    assert np.isclose(
        body.get_atmosperic_density(np.float64(0)), body.densitiy_at_surface * 100
    )
