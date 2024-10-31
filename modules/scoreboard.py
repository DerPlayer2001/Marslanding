import json

from lib.fetch import RequestHandler


class Scoreboard:
    """
    A class to represent a scoreboard.

    Attributes:
        url (str): The url of the scoreboard api
    """

    def __init__(self, url: str) -> None:
        """
        Initialize the scoreboard with the given url.

        Parameters:
            url (str): The url of the scoreboard api
        """
        self.url = url

    async def get_scoreboard(self) -> list[tuple[str, int]]:
        """
        Get the current scoreboard.

        Returns:
            list[tuple[str, int]]: The current scoreboard as a list of tuples with the player name and the score
        """
        res = await RequestHandler().get(self.url)
        return [(player, score) for player, score in json.loads(res)]

    async def add_score(self, player: str, score: int) -> None:
        """
        Add a score to the scoreboard.

        Parameters:
            player (str): The name of the player
            score (int): The score of the player
        """
        json = {"player": player, "score": score}
        await RequestHandler().post(self.url, data=json)
