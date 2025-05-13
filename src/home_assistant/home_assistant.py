from homeassistant_api import Client
from typing import List, Tuple, Union

from utils.color.color_utils import calculate_perceived_brightness

import requests
from urllib3.exceptions import InsecureRequestWarning


class HomeAssistant:
    """
    Class for interacting with Home Assistant
    """

    def __init__(self, url: str, access_token: str):
        """
        Initialize the Home Assistant connection.

        Args:
            url: Home Assistant API URL
            access_token: Long-lived access token for authentication
        """
        # Suppress HTTPS "localhost is insecure" warnings
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
        self.url = url
        self.access_token = access_token
        self._client = None

        # Living room light entity IDs
        self.clems_light = "light.clems_light"
        self.tv_light = "light.tv_light"
        self.living_room_lights = [self.clems_light, self.tv_light]

        self.last_brightness_pct: int = 100

    def _get_client(self) -> Client:
        """Get or create the Home Assistant API client."""
        if self._client is None:
            self._client = Client(self.url, self.access_token, verify_ssl=False)
        return self._client

    def dim_and_color(self, rgb):
        """Sends two requests: dims to 1 percent and changes color to rgb."""
        client = self._get_client()

        # Make sure to un-negate the r value before changing color
        rgb = [-rgb[0], rgb[1], rgb[2]]
        dim_pct = 1

        b = client.trigger_service(
            "light",
            "turn_on",
            entity_id=self.living_room_lights,
            brightness_pct=dim_pct,
        )
        c = client.trigger_service(
            "light",
            "turn_on",
            entity_id=self.living_room_lights,
            rgb_color=rgb,
        )
        self.last_brightness_pct = self._get_brightness_pct_from_light_state(c)
        return b, c

    def set_living_room_lights_color(
        self,
        rgb_color: Union[List[int], Tuple[int, int, int]],
        brightness_pct: int | None = None,
    ) -> None:
        """Set the color of both living room lights."""
        client = self._get_client()

        dark_scene = rgb_color[0] < 0
        if dark_scene:
            return self.dim_and_color(rgb_color)

        if brightness_pct is not None:
            return client.trigger_service(
                "light",
                "turn_on",
                entity_id=self.living_room_lights,
                rgb_color=rgb_color,
                brightness_pct=brightness_pct,
            )

    def _get_brightness_pct_from_light_state(self, light_state) -> int:
        try:
            format_255 = int(light_state[-1].attributes.get("brightness"))
            return int(format_255 / 255 * 100)
        except:
            return self.last_brightness_pct
