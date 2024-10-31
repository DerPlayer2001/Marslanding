import asyncio
from modules.body import Body
from modules.probe import Probe
from modules.scoreboard import Scoreboard
from modules.star import Star
from modules.types import Coordinate
import numpy as np
import pygame as pg
import lib.pygame_textinput
from enum import Enum, auto


class ChangeGameStatus(Enum):
    """
    Enum to represent if and how the game should change its status.
    """

    RESET = auto()
    EXIT = auto()
    NONE = auto()


class Game:
    """
    A class that represents the state of a game

    Attributes:
        body (Body): The Planet used
        probe (Probe): The Probe used
        scoreboard (Scoreboard): The current scoreboard
        landing_coord (np.float64): A Radius representing the position of the landing spot
        landed (Bool): Representing if the probe has landed
        font (pg.font.Font): The font used for Textoutputs
        screensize (Coordinate): The size of the Game screen
        screen (pg.surface.Surface): The Game screen
        clock (pg.time.Clock): The clock used for time intervals
        starship_pic (pg.surface.Surface): The picture of the starship as surface
        starship_dim (np.ndarray[int, dtype[int32]]): The dimensions of the Starship in pixels
        burn_pic (pg.surface.Surface): A picture of the fuel as surface
        burn_dim (np.ndarray[int, dtype[int32]]): The dimensions of the fuel-picture in pixels
        fuel_display_size (Coordinate): The dimensions of the fuel-display and the thrust-level-display
        stars (list[Star]): A list of stars
        radar_size (Coordinate): The dimensions of the radar
        height_limit (int): The altitude from wich the planets surface starts to show in meters
        scale (np.float64): The scale factor to Fit sizes of surfaces to the starship
        probe_size (int): The size of the probe on the radar in pixels
        score (int): The achieved score upon landing
        input (pg.surface.Surface): A surface that contains the text-input
        saved_name (Bool): Indicates if a name was put in
        input_done (Bool): Indicates if the input was terminated regardless if a name was provided
        original_probe (dict[str, Any]): The original probe for reset
        original_body (dict[str, Any]): The original body for reset
        original_scoreboard (dict[str, Any]): The original scoreboard for reset
        original_state (dict[str, Any]): The state of the game at the start for reset
    """

    def __init__(
        self,
        body: Body,
        probe: Probe,
        scoreboard: Scoreboard,
        landing_cord: np.float64,
        caption: str,
        starship_pic: pg.surface.Surface,
        burn_pic: pg.surface.Surface,
        screensize: Coordinate = (1300, 800),
        fuel_display_size: Coordinate = (40, 150),
        radar_size: Coordinate = (400, 200),
        height_limit: int = 200,
        probe_size: int = 5,
    ) -> None:
        """
        A function initializing a game with all the nessesary parameters and pygame functions

        Parameters:
            body (Body): An instance of a planet
            probe (Probe): An instance of a probe
            scoreboard (Scoreboard): The scoreboard to display and commit to
            landing_cord (np.float64): A Radius representing the position of the landing spot
            caption (str): The name of the game Window
            starship_pic (pg.surface.Surface): A picture of the starship
            burn_pic (pg.surface.Surface): A picture of the fuel exhaust
            screensize (Coordinate): The dimensions of the game window in pixels
            fuel_display_size (Coordinate): The dimensions of the fuel-display and the thrust-level-display in pixels
            radar_size (Coordinate): The dimensions of the radar in pixels
            height_limit (int): The altitude from wich the planets surface starts to show in meters
            probe_size (int): The radius of the circle representing the probe on radar
        """
        self.body = body
        self.probe = probe
        self.scoreboard = scoreboard
        self.landing_cord = landing_cord
        self.landed = False
        # pygame setup
        pg.init()
        pg.font.init()
        self.font = pg.font.Font(pg.font.get_default_font(), 20)
        self.screensize = screensize
        self.screen = pg.display.set_mode(screensize)
        pg.display.set_caption(caption)
        self.clock = pg.time.Clock()
        self.starship_pic, self.starship_dim = self.transform_surface(
            starship_pic, False
        )
        self.burn_pic, self.burn_dim = self.transform_surface(burn_pic, True)
        self.fuel_display_size = fuel_display_size
        self.stars = Star.generate_stars(100, screensize)
        self.radar_size = radar_size
        self.height_limit = height_limit
        self.scale = 50 / self.starship_dim[1]
        self.probe_size = probe_size
        self.score = 0

        self.input = self.get_surface((300, 50), pg.color.Color("White"))
        self.saved_name = False
        self.input_done = False

        # Save original state for reset
        self.original_probe = self.probe.__dict__.copy()
        self.original_body = self.body.__dict__.copy()
        self.original_scoreboard = self.scoreboard.__dict__.copy()
        self.original_state = self.__dict__.copy()

    async def rungame(self) -> None:
        """
        Contains the while loops that call all nessesary functions every tick
        """
        while not self.landed:
            await asyncio.sleep(0)  # to give timing control to the browser
            dt = np.float64(self.clock.tick(30) / 1000)
            self.process_tick(dt)
            match self.process_input(dt):
                case ChangeGameStatus.EXIT:
                    return self.exit()
                case ChangeGameStatus.RESET:
                    return await self.reset()
            self.screen.fill("black")
            vel = self.probe.get_rotated_vector(self.probe.velocity)
            Star.draw_stars(self.screen, self.stars, vel, dt, self.screensize)
            self.draw_starship()
            self.draw_body()
            self.draw_vel(vel)
            self.draw_arrow_to_landing()
            self.draw_fuel_display(
                pg.color.Color("red"),
                self.probe.fuel_mass / self.probe.max_fuel_mass,
                (10, 10),
            )
            self.draw_fuel_display(
                pg.color.Color("blue"),
                self.probe.get_burn_rate_percent(),
                (20 + self.fuel_display_size[0], 10),
            )
            self.draw_controls()
            self.draw_data_display()
            self.draw_radar()
            self.draw_controls()
            pg.display.flip()
        textinput = lib.pygame_textinput.TextInputVisualizer()
        scores = await self.scoreboard.get_scoreboard()
        while True:
            await asyncio.sleep(0)  # to give timing control to the browser
            match await self.draw_end_screen(textinput, scores):
                case ChangeGameStatus.EXIT:
                    return self.exit()
                case ChangeGameStatus.RESET:
                    return await self.reset()
            pg.display.flip()

    async def reset(self) -> None:
        """
        resets the Game to it's original state
        """
        new_state = self.original_state.copy()
        new_state["original_state"] = self.original_state
        self.__dict__ = new_state
        self.probe.__dict__ = self.original_probe.copy()
        self.body.__dict__ = self.original_body.copy()
        self.scoreboard.__dict__ = self.original_scoreboard.copy()
        await self.rungame()

    async def draw_end_screen(
        self,
        textinput: lib.pygame_textinput.TextInputVisualizer,
        scores: list[tuple[str, int]],
    ) -> ChangeGameStatus:
        """
        shows the score, the leaderboard and an input window to enter a name, to save score

        Parameters:
            textinput (lib.pygame_textinput.TextInputVisualizer): A textinput display
            scores (list[tuple[str, int]]): A list with all the scores with Name and score

        Returns:
            ChangeGameStatus
        """
        self.input.fill("grey")
        events = pg.event.get()
        for event in events:
            if event.type == pg.QUIT:
                return ChangeGameStatus.EXIT
            if event.type == pg.KEYDOWN:
                match event.key:
                    case pg.K_RETURN:
                        if not self.saved_name and len(textinput.value) > 0:
                            await self.save_score(textinput.value)
                            self.saved_name = True
                        self.input_done = True
                    case pg.K_r:
                        if self.input_done:
                            return ChangeGameStatus.RESET
        score_text = [
            self.get_text(f"{i+1}. {score}   {name}")
            for i, (name, score) in enumerate(scores[:5])
        ]
        if len(score_text) <= 0:
            score_text = [self.get_text("No Scores Yet")]
        score_screen = self.get_surface(
            (
                max([score[1][2] for score in score_text]) + 10,
                sum([score[1][3] for score in score_text]) + 30,
            ),
            pg.color.Color("black"),
            alpha=50,
        )
        counter = 5
        for score in score_text:
            score_screen.blit(score[0], (5, counter))
            counter += 5 + score[1][3]
        textinput.update(events)  # type: ignore
        self.input.blit(textinput.surface, (0, 0))
        input_sizex, input_sizey = self.input.get_rect()[2:]
        screenmidx, screenmidy = self.screensize[0] // 2, self.screensize[1] // 2
        if not self.input_done:
            self.screen.blit(
                self.input,
                (screenmidx - input_sizex // 2, screenmidy - input_sizey // 2 + 100),
            )
            self.screen.blit(
                self.get_text("Enter name to save score", "Black")[0],
                (
                    screenmidx - input_sizex // 2,
                    screenmidy - input_sizey // 2 + 100 + input_sizey,
                ),
            )
        else:
            pg.draw.rect(
                self.screen,
                self.body.color,
                (
                    screenmidx - input_sizex // 2,
                    screenmidy - input_sizey // 2 + 100,
                    input_sizex,
                    input_sizey + 50,
                ),
            )
        self.screen.blit(score_screen, (screenmidx - 205, 100))
        your_score = self.get_text(f"Your Score: {int(self.score)}", "black")
        your_landing_vel = self.get_text(
            f"Landing velocity: {round(np.linalg.norm(self.probe.velocity),2)}m/s",
            "black",
        )
        your_distance = self.get_text(
            f"Distance to Target: {round(np.linalg.norm(self.get_landingspot_to_ship_vector()),2)}m",
            "black",
        )
        your_tilt = self.get_text(
            f"Tilt: {round(self.probe.get_tilt_error(), 2)}°", "black"
        )
        survived = (
            self.get_text("Your probe survived", "black")
            if self.probe.is_successful_landing()
            else self.get_text("Your probe didn't survive", "black")
        )
        fuel_left = self.get_text(f"Fuel_Left: {self.probe.fuel_mass}kg", "black")
        stats = [
            your_score,
            your_landing_vel,
            your_distance,
            your_tilt,
            survived,
            fuel_left,
        ]
        stat_screen = self.get_surface(
            (max([stat[1][2] for stat in stats]), sum([stat[1][3] for stat in stats])),
            self.body.color,
        )
        counter = 0
        for stat in stats:
            stat_screen.blit(stat[0], (0, counter))
            counter += stat[1][3] + 5
        self.screen.blit(
            stat_screen,
            (
                self.screensize[0] - stat_screen.get_rect()[2] - 5,
                self.screensize[1] - stat_screen.get_rect()[3] - 5,
            ),
        )
        return ChangeGameStatus.NONE

    def draw_controls(self) -> None:
        """draws a screen on the bottom to show the controls"""
        texts = [
            self.get_text("A/D: Rotate"),
            self.get_text("W/S: Regulate Thrust"),
            self.get_text("Q/E: No/Full Thrust"),
            self.get_text("R: Reset Game"),
        ]
        size = [(x, y) for _, [_, _, x, y] in texts]
        screen1 = self.get_surface(
            (max(size[0][0], size[1][0] + 10), size[0][1] + size[1][1] + 10)
        )
        screen2 = self.get_surface(
            (max(size[0][0], size[1][0] + 10), size[0][1] + size[1][1] + 10)
        )
        _, screeny = self.screensize
        screen1.blit(texts[0][0], (5, 5))
        screen1.blit(texts[1][0], (5, 10 + texts[0][1][3]))
        screen2.blit(texts[2][0], (5, 5))
        screen2.blit(texts[3][0], (5, 10 + texts[2][1][3]))
        screen1x, screen1y = screen1.get_rect()[2:]
        self.screen.blit(screen1, (200, screeny - screen1y - 20))
        self.screen.blit(screen2, (220 + screen1x, screeny - screen1y - 20))

    def exit(self) -> None:
        """Ends the game and closes the window"""
        pg.display.quit()
        pg.quit()

    def get_landingspot_coords(self) -> np.ndarray[np.float64, np.dtype[np.float64]]:
        """
        calculates the coordinates of the landing spot

        Returns:
            landingspot (np.ndarray[np.float64, np.dtype[np.float64]]): The coordinates of the landing spot
        """
        landingspot = np.array([np.cos(self.landing_cord), np.sin(self.landing_cord)])
        return landingspot * self.body.radius

    def get_landingspot_to_ship_vector(
        self,
    ) -> np.ndarray[np.float64, np.dtype[np.float64]]:
        """calculates the vector from ship to landingspot on the rotated screen

        Returns:
            rotated_vector (np.ndarray[np.float64, np.dtype[np.float64]]): The vector from ship to landing-spot
        """
        ship_spot_vec = self.get_landingspot_coords() - self.probe.position
        return self.probe.get_rotated_vector(ship_spot_vec)

    def draw_arrow_to_landing(self) -> None:
        """draws a line to indicate where the landing spot is"""
        vector = self.get_landingspot_to_ship_vector()
        vectorx, vectory = vector / np.linalg.norm(vector)
        _, starshipdimy = self.starship_dim * self.scale
        screenx, screeny = self.screensize
        screenmidx, screenmidy = screenx // 2, screeny // 2
        pg.draw.line(
            self.screen,
            "red",
            (
                vectorx * (starshipdimy + 10) + screenmidx,
                -vectory * (starshipdimy + 10) + screenmidy,
            ),
            (
                vectorx * (starshipdimy + 10 + 50) + screenmidx,
                -vectory * (starshipdimy + 10 + 50) + screenmidy,
            ),
        )

    def draw_body(self) -> None:
        """draws the planet surface when it is close"""
        altitude = self.get_altitude()
        screen_x, screen_y = self.screensize
        screenmid_x, screenmid_y = screen_x // 2, screen_y // 2
        _, starship_dim_y = self.starship_dim * self.scale
        bottom_starship = screenmid_y + starship_dim_y // 2
        if altitude < self.height_limit:
            pg.draw.rect(
                self.screen,
                self.body.color,
                (
                    0,
                    int(
                        bottom_starship
                        + (screenmid_y - starship_dim_y // 2)
                        * (altitude / self.height_limit)
                    ),
                    screen_x,
                    int(screenmid_x),
                ),
            )
            spotx, _ = 2 * self.get_landingspot_to_ship_vector()
            size = 20
            pg.draw.rect(
                self.screen,
                pg.color.Color("red"),
                (
                    spotx - size + screenmid_x,
                    bottom_starship
                    + (screenmid_y - starship_dim_y // 2)
                    * (altitude / self.height_limit),
                    2 * size,
                    10,
                ),
            )

    def draw_data_display(self) -> None:
        """adds a data display in the top-left corner to show important data"""
        display_Texts = [
            self.get_text(f"Speed: {round(np.linalg.norm(self.probe.velocity),2)} m/s"),
            self.get_text(f"Altitude: {round(self.get_altitude(), 2)} m"),
            self.get_text(f"Fuel: {round(self.probe.fuel_mass, 2)} kg"),
            self.get_text(f"Tilt: {round(self.probe.get_tilt_error(), 2)}°"),
            self.get_text(
                f"ToTarget: {int(np.linalg.norm(self.get_landingspot_to_ship_vector()))} m"
            ),
        ]

        data_Display = self.get_surface(
            (
                max([x[1][2] for x in display_Texts]) + 10,
                sum([y[1][3] for y in display_Texts]) + 10 * len(display_Texts),
            )
        )
        text_position = 10
        for text in display_Texts:
            data_Display.blit(text[0], (10, text_position))
            text_position += 10 + text[1][3]

        self.screen.blit(data_Display, (0, 0))

    def draw_starship(self) -> None:
        """draws the spaceship with the correctly sized exhaust and the right rotation"""
        starship_dim_x, starship_dim_y = self.starship_dim
        screen_x, screen_y = self.screensize
        screenmid_x, screenmid_y = screen_x // 2, screen_y // 2
        _, burn_dim_y = self.burn_dim
        starship_size_x, starship_size_y = (
            starship_dim_x,
            starship_dim_y + 2 * burn_dim_y,
        )
        starship = pg.surface.Surface((starship_size_x, starship_size_y), pg.SRCALPHA)
        starship.blit(
            self.starship_pic,
            (
                starship_size_x // 2 - starship_dim_x // 2,
                starship_size_y // 2 - starship_dim_y // 2,
            ),
        )

        if self.probe.current_burn_rate > 0 and self.probe.fuel_mass > 0:
            fuel_Pic_Scaled = pg.transform.scale_by(
                self.burn_pic, float(self.probe.get_burn_rate_percent())
            )
            new_fuel_pic_dim_x, _ = fuel_Pic_Scaled.get_rect()[2:]
            starship.blit(
                fuel_Pic_Scaled,
                (
                    starship_size_x // 2 - new_fuel_pic_dim_x // 2,
                    starship_size_y // 2 + starship_dim_y // 2,
                ),
            )
        starship = pg.transform.scale_by(starship, 50 / starship_dim_y)

        rotated_starship = pg.transform.rotate(starship, float(self.probe.get_tilt()))
        size_x, size_y = rotated_starship.get_rect()[2:]
        self.screen.blit(
            rotated_starship, (screenmid_x - size_x // 2, screenmid_y - size_y // 2)
        )

    def draw_vel(self, vel: np.ndarray[np.float64, np.dtype[np.float64]]) -> None:
        """draws a line to indicate the current velocity-direction"""
        velx, vely = vel
        _, starship_dim_y = self.starship_dim * self.scale
        screen_x, screen_y = self.screensize
        screenmid_x, screenmid_y = screen_x // 2, screen_y // 2
        pg.draw.line(
            self.screen,
            "white",
            (screenmid_x, screenmid_y - starship_dim_y),
            (screenmid_x, screenmid_y - vely - starship_dim_y),
        )
        pg.draw.line(
            self.screen,
            "white",
            (screenmid_x, screenmid_y - starship_dim_y),
            (screenmid_x + velx, screenmid_y - starship_dim_y),
        )

    def draw_radar(self) -> None:
        """adds a radar in the top right corner that shows the curent probe-position in relation to the planet"""
        radar = self.get_surface(self.radar_size, pg.color.Color("grey"))
        size_x, size_y = np.array(self.radar_size)
        screen_x, _ = self.screensize
        buffer = self.probe_size * 2
        body_radius = float(
            self.body.radius
            * (min(size_y, size_x) // 2 - buffer)
            / np.linalg.norm(self.probe.position)
        )
        radar_probe_distance_x, radar_probe_distance_y = (
            self.probe.position
            * (min(size_x, size_y) // 2 - buffer)
            / (np.linalg.norm(self.probe.position))
        )
        pg.draw.circle(
            radar,
            self.body.color,
            (size_x // 2, size_y // 2),
            body_radius,
        )

        pg.draw.line(
            radar,
            "blue",
            (size_x // 2, size_y // 2),
            (
                size_x // 2 + body_radius * np.cos(self.landing_cord),
                size_y // 2 + body_radius * np.sin(self.landing_cord),
            ),
        )

        pg.draw.circle(
            radar,
            "red",
            (
                int(radar_probe_distance_x + size_x // 2),
                int(-radar_probe_distance_y + size_y // 2),
            ),
            self.probe_size,
        )
        self.screen.blit(radar, (screen_x - size_x, 0))

    def draw_fuel_display(
        self,
        color: pg.color.Color,
        percentage: np.float64,
        position: Coordinate,
    ) -> None:
        """
        adds two bars in the bottom left corner that display the leftover fuel and the current thrust level

        Parameters:
            color (pg.color.Color): The color of the display bar
            percentage (np.float64): The percentage of the bar that is filled
            position (Coordinate): The posiotion on the screen
        """
        size_x, size_y = self.fuel_display_size
        _, screen_y = self.screensize
        px, py = position
        fuel_display = self.get_surface(self.fuel_display_size, pg.color.Color("grey"))
        pg.draw.rect(
            fuel_display,
            color,
            (
                size_x * 0.2 / 2,
                5,
                size_x * 0.8,
                size_y - 10,
            ),
        )

        pg.draw.rect(
            fuel_display,
            "black",
            (
                size_x * 0.2 / 2,
                5,
                size_x * 0.8,
                int((1 - percentage) * (size_y - 10)),
            ),
        )

        self.screen.blit(fuel_display, (px, screen_y - size_y - py))

    def process_input(self, dt: np.float64) -> ChangeGameStatus:
        """
        handles all input that was done during a tick

        Parameters:
            dt (np.float64): the time since the last tick

        Returns:
            ChangeGameStatus
        """
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return ChangeGameStatus.EXIT
        keys = pg.key.get_pressed()
        if keys[pg.K_a]:
            self.probe.orientation = (self.probe.orientation + 2 * dt) % (2 * np.pi)
        if keys[pg.K_d]:
            self.probe.orientation = (self.probe.orientation - 2 * dt) % (2 * np.pi)
        if keys[pg.K_q]:
            self.probe.set_burn_rate(np.float64(0))
        if keys[pg.K_e]:
            self.probe.set_burn_rate(np.float64(1))
        if keys[pg.K_r]:
            return ChangeGameStatus.RESET
        if keys[pg.K_s]:
            self.probe.change_burn_rate(-dt)
        if keys[pg.K_w]:
            self.probe.change_burn_rate(dt)
        return ChangeGameStatus.NONE

    def transform_surface(
        self, surface: pg.surface.Surface, factor: bool
    ) -> tuple[pg.surface.Surface, np.ndarray[int, np.dtype[np.int32]]]:
        """
        transforms the relevant surface to a size that fits the spaceship

        Parameters:
            surface (pg.surface.Surface): The surface to resize
            factor (bool): Determines if a scalefactor is used

        Returns:
            tuple[surface, np.array(surface.get_rect()[2:])]: A tuple with the resized surface and its size
        """
        surface_dim = surface.get_rect()[2:]
        if factor:
            surface = pg.transform.scale_by(
                surface, self.starship_dim[1] / surface_dim[1]
            )
        return surface, np.array(surface.get_rect()[2:])

    def process_tick(self, dt: np.float64) -> None:
        """
        checks if the probe has landed and calculates values to update the probe if it hasn't

        Parameters:
            dt (np.float64): The time since the last tick
        """
        if self.landed:
            return
        self.probe.process_tick(self.body, dt)
        px, py = self.probe.position
        theta = np.arctan2(py, px)
        if np.linalg.norm(self.probe.position) <= self.body.radius:
            self.calculate_score()
            self.probe.position = (
                np.array([np.cos(theta), np.sin(theta)]) * self.body.radius
            )
            self.probe.current_burn_rate = 0
            self.landed = True

    def calculate_score(self) -> None:
        """Calculates a score based on speed, tilt and offset to target"""
        if not self.landed:
            self.score = 0
        offset = np.linalg.norm(self.get_landingspot_to_ship_vector())
        speed = np.linalg.norm(self.probe.velocity)
        tilt = np.abs(self.probe.get_tilt_error())
        used_fuel = self.probe.max_fuel_mass - self.probe.fuel_mass

        print(f"Landed {offset} off target at {speed} m/s.")
        offset_score = 1000 / (offset + 1000)
        speed_score = 1000 / (speed + 1000)
        tilt_score = 1000 / (tilt + 1000)
        fuel_score = 1000 / (used_fuel + 1000)
        survival_score = 1000 if self.probe.is_successful_landing() else 500
        self.score = (
            fuel_score * offset_score * speed_score * tilt_score * survival_score
        )

    async def save_score(self, name: str) -> None:
        """
        saves a score to the scoreboard

        Parameters:
            name (str): The name of the player
        """
        await self.scoreboard.add_score(name, int(self.score))

    def get_text(
        self, text: str, color: str = "white"
    ) -> tuple[pg.surface.Surface, pg.rect.Rect]:
        """
        turns a string into a surface with text and it's size

        Parameters:
            text (str): The text to display
            color (str): color of the text

        Returns:
            tuple[pg.surface.Surface, pg.rect.Rect]: A tuple with the Text-surface and it's size
        """
        out = self.font.render(text, True, color)
        return (out, out.get_rect())

    def get_altitude(self) -> np.float64:
        """
        calculates the altitude of the probe in meters

        Returns:
            altitude (np.float64): Distance from starship to planet-surface
        """
        return self.probe.get_altitude(self.body)

    def get_surface(
        self,
        size: Coordinate,
        color: pg.color.Color | None = None,
        alpha: int | None = None,
    ) -> pg.surface.Surface:
        """
        A helper function to get pg.Surfaces

        Parameters:
            size (Coordinate): The dimensions of the surface
            color (pg.color.Color | None): The color of the surface
            alpha (int | None): The transparency of the Surface

        Returns:
            surface (pg.surface.Surface): A surface used for displaying
        """
        surface = pg.surface.Surface(size)
        if color is not None:
            surface.fill(color)
        if alpha is not None:
            surface.set_alpha(alpha)
        return surface


class DefaultMarsGame(Game):
    """
    A default game with realistic Mars parameters
    """

    def __init__(self, backend_url: str) -> None:
        """
        initializes a DefaultMarsGame

        Parameters:
            backend_url (str): The url of the scoreboard api
        """
        body = Body(  # (6)
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
            np.array([body.radius + 1000, -1000]),
            np.array([0, 100]),
            np.float64(-np.pi / 2),
            np.float64(1.7),
            np.float64(25),
            np.float64(5),
            np.float64(15),
        )
        scoreboard = Scoreboard(backend_url)
        starship_pic = pg.image.load("assets/spaceship_rect.png")
        fuel_Pic = pg.image.load("assets/Fire.png")
        super().__init__(
            body,
            probe,
            scoreboard,
            np.float64(0),
            "Marslandung",
            starship_pic,
            fuel_Pic,
        )


class TestEarthGeostationary(Game):
    """
    A default Game to simulate a geostationary orbit around Earth
    """

    def __init__(self, backend_url: str) -> None:
        """
        initializes a TestEarthGeostationary Game

        Parameters:
            backend_url (str): The url of the scoreboard api
        """
        body = Body(  # (6)
            np.float64(5.9742e24),
            np.float64(6371000),
            pg.color.Color("blue"),
            np.float64(1000),
            np.float64(6.5e-3),
            np.float64(288.15),
            np.float64(28.96e-3),
        )
        probe = Probe(
            np.float64(500),
            np.float64(1000),
            np.float64(10),
            np.float64(1000),
            np.array([0, 42164000]),
            np.array([-3066, 0]),
            np.float64(-np.pi / 2),
            np.float64(1.7),
            np.float64(4),
            np.float64(5),
            np.float64(15),
        )
        scoreboard = Scoreboard(backend_url)
        starship_pic = pg.image.load("assets/spaceship_rect.png")
        fuel_Pic = pg.image.load("assets/Fire.png")
        super().__init__(
            body,
            probe,
            scoreboard,
            np.float64(0),
            "Erde geostationär",
            starship_pic,
            fuel_Pic,
        )
