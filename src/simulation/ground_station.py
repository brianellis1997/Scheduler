"""
Ground Station Management Module

Handles ground station visibility windows and data downlink.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class GroundStation:
    """Ground station definition"""

    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_deg: float = 10.0  # Minimum elevation angle for contact
    max_bandwidth_mbps: float = 50.0


class GroundStationManager:
    """
    Manages ground station contacts and data downlink

    Simplified visibility model based on elevation angle and distance.
    """

    def __init__(self, stations: Optional[List[GroundStation]] = None):
        """
        Initialize ground station manager

        Args:
            stations: List of GroundStation objects. If None, uses default stations.
        """
        if stations is None:
            # Default ground stations (real example locations)
            stations = [
                GroundStation("Fairbanks", latitude_deg=64.8, longitude_deg=-147.7),
                GroundStation("Santiago", latitude_deg=-33.4, longitude_deg=-70.4),
                GroundStation("Singapore", latitude_deg=1.4, longitude_deg=103.8),
            ]

        self.stations = stations
        self.max_communication_distance_km = 4000  # Approximate line-of-sight distance

    def check_visibility(self, sat_latitude_deg: float, sat_longitude_deg: float,
                        sat_altitude_km: float = 500) -> List[Tuple[str, bool]]:
        """
        Check which ground stations are visible from satellite position

        Args:
            sat_latitude_deg: Satellite latitude in degrees
            sat_longitude_deg: Satellite longitude in degrees
            sat_altitude_km: Satellite altitude in km

        Returns:
            List of (station_name, is_visible) tuples
        """
        visibility = []

        for station in self.stations:
            # Compute great circle distance (simplified)
            lat_diff = sat_latitude_deg - station.latitude_deg
            lon_diff = sat_longitude_deg - station.longitude_deg

            # Simple distance approximation (not geodetically accurate)
            distance_deg = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
            distance_km = distance_deg * 111.0  # ~111 km per degree

            # Compute elevation angle (simplified geometry)
            earth_radius_km = 6371.0
            distance_2d = np.sqrt((distance_km) ** 2)

            if distance_2d > 0:
                elevation_rad = np.arctan(sat_altitude_km / distance_2d) - np.arcsin(
                    earth_radius_km / (earth_radius_km + sat_altitude_km)
                )
                elevation_deg = np.degrees(elevation_rad)
            else:
                elevation_deg = 90.0

            # Check if elevation exceeds minimum
            is_visible = elevation_deg >= station.elevation_deg

            visibility.append((station.name, is_visible))

        return visibility

    def downlink_data(self, data_stored_gb: float, downlink_duration_s: float,
                     available_bandwidth_mbps: float = 50.0) -> Tuple[float, float]:
        """
        Simulate data downlink to ground station

        Args:
            data_stored_gb: Amount of data stored on satellite in GB
            downlink_duration_s: Duration of downlink window in seconds
            available_bandwidth_mbps: Available bandwidth in Mbps

        Returns:
            (data_downlinked_gb, remaining_data_gb)
        """
        # Compute max data that can be downlinked
        data_rate_gbps = available_bandwidth_mbps / 8000.0  # Convert Mbps to GBps
        max_downlink_gb = data_rate_gbps * downlink_duration_s

        # Downlink is limited by available data and bandwidth
        data_downlinked = min(data_stored_gb, max_downlink_gb)

        remaining_data = data_stored_gb - data_downlinked

        return data_downlinked, remaining_data

    def predict_next_contact(self, current_epoch_s: float,
                            orbital_period_s: float) -> Tuple[str, float]:
        """
        Predict next ground station contact

        Args:
            current_epoch_s: Current simulation epoch in seconds
            orbital_period_s: Orbital period in seconds

        Returns:
            (station_name, time_to_next_contact_s)
        """
        # Simplified: predict contact with first station after ~40% of orbit
        time_to_next = orbital_period_s * 0.4
        station_name = self.stations[0].name

        return station_name, time_to_next

    def get_contact_window_duration(self) -> float:
        """
        Estimate typical ground station contact window duration

        Returns:
            Contact window duration in seconds
        """
        # Typical LEO satellite pass duration is 10-15 minutes
        return np.random.uniform(600, 900)
