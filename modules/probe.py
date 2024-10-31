import numpy as np
from modules.body import Body
import modules.constants as constants


class Probe:
    """
    A class to represent a probe.

    Attributes:
        dry_mass (np.float64): The dry mass of the probe in kg
        fuel_mass (np.float64): The mass of the fuel in the probe in kg
        max_fuel_mass (np.float64): The maximum mass of the fuel in the probe in kg
        max_burn_rate (np.float64): The maximum burn rate of the probe in kg/s
        specific_impulse (np.float64): The Specific Impulse in N * s / kg
        position (np.ndarray[np.float64, np.dtype[np.float64]]): The position of the probe in m
        velocity (np.ndarray[np.float64, np.dtype[np.float64]]): The velocity of the probe in m/s
        orientation (np.float64): The orientation of the probe in rad
        current_burn_rate (np.float64): The current burn rate of the probe in kg/s
        drag_coefficient (np.float64): The drag coefficient of the probe
        surface_area (np.float64): The surface area of the probe in m^2
        landing_speed_tolerance (np.float64): The tolerance for the landing speed of the probe in m/s
        landing_tilt_tolerance (np.float64): The tolerance for the landing tilt of the probe in degrees
    """

    def __init__(
        self,
        dry_mass: np.float64,
        fuel_mass: np.float64,
        max_burn_rate: np.float64,
        specific_impulse: np.float64,
        position: np.ndarray[np.float64, np.dtype[np.float64]],
        velocity: np.ndarray[np.float64, np.dtype[np.float64]],
        orientation: np.float64,
        drag_coefficient: np.float64,
        surface_area: np.float64,
        landing_speed_tolerance: np.float64,
        landing_tilt_tolerance: np.float64,
    ) -> None:
        """
        Initialize the probe with the given parameters.

        Parameters:
            dry_mass (np.float64): The dry mass of the probe in kg
            fuel_mass (np.float64): The mass of the fuel in the probe in kg
            max_burn_rate (np.float64): The maximum burn rate of the probe in kg/s
            specific_impulse (np.float64): The Specific Impulse in N * s/ kg
            position (np.ndarray[np.float64, np.dtype[np.float64]]): The position of the probe in m
            velocity (np.ndarray[np.float64, np.dtype[np.float64]]): The velocity of the probe in m/s
            orientation (np.float64): The orientation of the probe in rad
            drag_coefficient (np.float64): The drag coefficient of the probe in kg / m
            surface_area (np.float64): The surface area of the probe in m^2
            landing_speed_tolerance (np.float64): The tolerance for the landing speed of the probe in m/s
            landing_tilt_tolerance (np.float64): The tolerance for the landing tilt of the probe in degrees
        """
        if dry_mass <= 0:
            raise ValueError("Dry mass must be greater than 0")
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass
        self.max_fuel_mass = fuel_mass
        self.max_burn_rate = max_burn_rate
        self.specific_impulse = specific_impulse
        self.position = position
        self.velocity = velocity
        self.orientation = orientation
        self.current_burn_rate = 0
        self.drag_coefficient = drag_coefficient
        self.surface_area = surface_area
        self.landing_speed_tolerance = landing_speed_tolerance
        self.landing_tilt_tolerance = landing_tilt_tolerance

    def is_successful_landing(self) -> bool:
        """
        Check if the probe has successfully landed.

        Returns:
            bool: True if the probe has successfully landed, False otherwise
        """
        if np.linalg.norm(self.velocity) > self.landing_speed_tolerance:
            return False
        if abs(self.get_tilt_error()) > self.landing_tilt_tolerance:
            return False
        return True

    def get_mass(self) -> np.float64:
        """
        Calculate the mass of the probe.

        Returns:
            np.float64: The mass of the probe in kg
        """
        return self.dry_mass + self.fuel_mass

    def process_burn(self, dt: np.float64) -> np.float64:
        """
        Process the burn of the probe.
        uses the specific impulse (5)

        Parameters:
            dt (np.float64): The time step of the simulation in s

        Returns:
            np.float64: A Factor representing the Integral over the time step in respect to the change in mass in s / kg
        """
        self.current_burn_rate = min(
            self.max_burn_rate, self.fuel_mass / dt, self.current_burn_rate
        )
        m0 = self.get_mass()

        dm = self.current_burn_rate * dt

        if dm == 0:
            # Integral only works if dm is not 0
            # But in this case we don't even need an integral
            return dt / m0

        self.fuel_mass -= dm

        factor = dt * np.log(m0 / (m0 - dm)) / dm

        F = self.current_burn_rate * self.specific_impulse

        self.velocity = self.velocity - (
            np.array(
                [
                    np.cos(self.orientation),
                    np.sin(self.orientation),
                ]
            )
            * F
            * factor
        )

        return factor

    def get_altitude(self, body: Body) -> np.float64:
        """
        Calculate the altitude of the probe above a given body.

        Parameters:
            body (Body): The body above which the altitude should be calculated

        Returns:
            np.float64: The altitude of the probe above the body in m
        """
        return np.linalg.norm(self.position) - body.radius

    def process_drag(self, body: Body, factor: np.float64) -> None:
        """
        Process the drag of the probe.
        uses the drag force (4)

        Parameters:
            body (Body): The body in the atmosphere of which the probe is
            factor (np.float64): A Factor representing the Integral over the time step in respect to the change in mass in s / kg
        """
        altitude = self.get_altitude(body)
        atmospheric_density = (
            body.get_atmosperic_density(altitude) * 1e-5
        )  # convert to bar
        vx, vy = self.velocity
        theta = np.arctan2(vy, vx)
        F = (
            0.5
            * atmospheric_density
            * np.linalg.norm(self.velocity) ** 2
            * self.drag_coefficient
            * self.surface_area
        )
        self.velocity = self.velocity - (
            np.array(
                [
                    np.cos(theta),
                    np.sin(theta),
                ]
            )
            * F
            * factor
        )

    def process_movement(self, body: Body, dt: np.float64, factor: np.float64) -> None:
        """
        Process the movement of the probe.
        uses the gravitational force (3)

        Parameters:
            body (Body): The body around which the probe is
            dt (np.float64): The time step of the simulation in s
            factor (np.float64): A Factor representing the Integral over the time step in respect to the change in mass in s / kg
        """
        G = constants.GRAVITATIONAL_CONSTANT
        r = np.linalg.norm(self.position)
        px, py = self.position
        theta = np.arctan2(py, px)

        F = G * body.mass * self.get_mass() / r**2

        self.velocity = self.velocity - (
            np.array(
                [
                    np.cos(theta),
                    np.sin(theta),
                ]
            )
            * F
            * factor
        )

        self.position = self.position + self.velocity * dt

    def set_burn_rate(self, percent: np.float64) -> None:
        """
        Set the burn rate of the probe.

        Parameters:
            percent (np.float64): The percentage of the maximum burn rate
        """
        self.current_burn_rate = percent * self.max_burn_rate

    def get_burn_rate_percent(self) -> np.float64:
        """
        Get the burn rate of the probe as a percentage of the maximum burn rate.

        Returns:
            np.float64: The percentage of the maximum burn rate
        """
        return self.current_burn_rate / self.max_burn_rate

    def change_burn_rate(self, percent: np.float64) -> None:
        """
        Change the burn rate of the probe by a given percentage.

        Parameters:
            percent (np.float64): The percentage by which the burn rate should be changed
        """
        current_percent = self.get_burn_rate_percent()
        new_percent = current_percent + percent
        if new_percent < 0:
            new_percent = 0
        if new_percent > 1:
            new_percent = 1
        self.set_burn_rate(np.float64(new_percent))

    def process_tick(self, body: Body, dt: np.float64) -> None:
        """
        Process a tick of the simulation.

        Parameters:
            body (Body): The body around which the probe is
            dt (np.float64): The time step of the simulation
        """
        factor = self.process_burn(dt)
        self.process_drag(body, factor)
        self.process_movement(body, dt, factor)

    def get_tilt(self) -> np.float64:
        """
        Get the tilt of the probe.

        Returns:
            np.float64: The tilt of the probe in degrees
        """
        px, py = self.position
        angle1 = np.arctan2(py, px)
        orientation = self.orientation
        return (((np.pi - angle1 + orientation) % (2 * np.pi)) / np.pi) * 180

    def get_rotated_vector(
        self,
        vector: np.ndarray[np.float64, np.dtype[np.float64]],
    ) -> np.ndarray[np.float64, np.dtype[np.float64]]:
        """
        Get a vector rotated by the orientation of the probe.

        Parameters:
            vector (np.ndarray[np.float64, np.dtype[np.float64]]): The vector to rotate

        Returns:
            np.ndarray[np.float64, np.dtype[np.float64]]: The rotated vector
        """
        px, py = self.position
        angle = np.pi / 2 - np.arctan2(py, px)
        rotated_vector = np.matmul(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            vector,
        )
        return rotated_vector

    def get_tilt_error(self) -> np.float64:
        return 180 - ((self.get_tilt() + 180) % 360)


def test() -> None:
    import pygame as pg

    """
    Run tests for the Probe class.
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
    probe = Probe(
        np.float64(250),
        np.float64(100),
        np.float64(10),
        np.float64(1000),
        np.array([body.radius + 1000, 0]),
        np.array([0, 100]),
        np.float64(-np.pi / 2),
        np.float64(1.7),
        np.float64(25),
        np.float64(5),
        np.float64(15),
    )
    copy = probe.__dict__.copy()
    probe.process_drag(body, 1 / probe.get_mass())
    assert np.linalg.norm(probe.velocity) < np.linalg.norm(copy["velocity"])
    probe.velocity = np.array([0, 0])
    probe.process_movement(body, np.float64(1), 1 / probe.get_mass())
    assert np.linalg.norm(probe.velocity) > 0
    assert np.linalg.norm(probe.position) < np.linalg.norm(copy["position"])
    probe.velocity = np.array([0, 0])
    probe.set_burn_rate(np.float64(1))
    probe.process_burn(np.float64(1))
    assert np.linalg.norm(probe.velocity) > 0
