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

        self.last_brightness_pct: int = 100

    def _get_client(self) -> Client:
        """Get or create the Home Assistant API client."""
        if self._client is None:
            self._client = Client(self.url, self.access_token, verify_ssl=False)
        return self._client

    # Individual light control methods
    def turn_on_light(self, entity_id: str) -> None:
        """Turn on a specific light."""
        client = self._get_client()
        client.trigger_service("light", "turn_on", entity_id=entity_id)

    def turn_off_light(self, entity_id: str) -> None:
        """Turn off a specific light."""
        client = self._get_client()
        client.trigger_service("light", "turn_off", entity_id=entity_id)

    def set_light_color(
        self,
        entity_id: str,
        rgb_color: Union[List[int], Tuple[int, int, int]],
        brightness_pct: int | None = None,
    ) -> None:
        """
        Set the color of a specific light.

        Args:
            entity_id: The entity ID of the light
            rgb_color: RGB color as a list or tuple of 3 integers (0-255)
        """
        if brightness_pct is None:
            brightness_pct = min(
                int(calculate_perceived_brightness(rgb_color) * 2.5), 100
            )

        client = self._get_client()
        client.trigger_service(
            "light",
            "turn_on",
            entity_id=entity_id,
            rgb_color=rgb_color,
            brightness_pct=brightness_pct,
        )

    # Clem's light
    def turn_on_clems_light(self) -> None:
        """Turn on Clem's light."""
        self.turn_on_light(self.clems_light)

    def turn_off_clems_light(self) -> None:
        """Turn off Clem's light."""
        self.turn_off_light(self.clems_light)

    def set_clems_light_color(
        self,
        rgb_color: List[int] | Tuple[int, int, int],
        brightness_pct: None | int,
    ) -> None:
        """Set the color of Clem's light."""
        self.set_light_color(self.clems_light, rgb_color, brightness_pct)

    # TV light
    def turn_on_tv_light(self) -> None:
        """Turn on the TV light."""
        self.turn_on_light(self.tv_light)

    def turn_off_tv_light(self) -> None:
        """Turn off the TV light."""
        self.turn_off_light(self.tv_light)

    def set_tv_light_color(
        self,
        rgb_color: List[int] | Tuple[int, int, int],
        brightness_pct: None | int,
    ) -> None:
        """Set the color of the TV light."""
        self.set_light_color(self.tv_light, rgb_color, brightness_pct)

    # Combined living room light methods
    def turn_on_living_room_lights(self) -> None:
        """Turn on both living room lights."""
        client = self._get_client()
        client.trigger_service(
            "light",
            "turn_on",
            entity_id=[self.clems_light, self.tv_light],
        )

    def turn_off_living_room_lights(self) -> None:
        """Turn off both living room lights."""
        client = self._get_client()
        client.trigger_service(
            "light",
            "turn_off",
            entity_id=[self.clems_light, self.tv_light],
        )

    def dim_and_color(self, rgb):
        """Sends two requests: dims to 1 percent and changes color to rgb."""
        # Make sure to un-negate the r value before changing color
        client = self._get_client()

        rgb = [-rgb[0], rgb[1], rgb[2]]
        dim_pct = 1

        b = client.trigger_service(
            "light",
            "turn_on",
            entity_id=[self.clems_light, self.tv_light],
            brightness_pct=dim_pct,
        )
        c = client.trigger_service(
            "light",
            "turn_on",
            entity_id=[self.clems_light, self.tv_light],
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
                entity_id=[self.clems_light, self.tv_light],
                rgb_color=rgb_color,
                brightness_pct=brightness_pct,
            )

    def _get_brightness_pct_from_light_state(self, light_state) -> int:
        try:
            format_255 = int(light_state[-1].attributes.get("brightness"))
            return int(format_255 / 255 * 100)
        except:
            return self.last_brightness_pct
