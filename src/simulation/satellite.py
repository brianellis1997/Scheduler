"""
Satellite State and Dynamics Module

Manages satellite orbital state, attitude, power, and memory.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SatelliteState:
    """Container for satellite state variables"""

    # Orbital elements (simplified)
    epoch: float
    altitude_km: float
    inclination_deg: float
    longitude_deg: float
    latitude_deg: float

    # Attitude
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    yaw_deg: float = 0.0

    # Power system
    battery_soc: float = 1.0  # State of Charge [0, 1]
    battery_capacity_wh: float = 100.0
    power_generation_w: float = 0.0

    # Data storage
    data_storage_gb: float = 0.0
    storage_capacity_gb: float = 10.0

    # Thermal/other constraints
    temperature_c: float = 20.0


class SatelliteModel:
    """
    Satellite dynamics model for LEO observation satellite

    Simplified model for RL training:
    - Uses analytical orbit propagation
    - Basic power budget model (solar + battery)
    - Simple memory accumulation model
    - Realistic slew constraints
    """

    def __init__(self, altitude_km=500, inclination_deg=97.5, max_slew_rate_deg_s=5.0):
        """
        Initialize satellite model

        Args:
            altitude_km: Orbital altitude in km
            inclination_deg: Orbital inclination in degrees
            max_slew_rate_deg_s: Maximum attitude slew rate in deg/s
        """
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg
        self.max_slew_rate_deg_s = max_slew_rate_deg_s

        # Orbital mechanics constants
        self.earth_radius_km = 6371.0
        self.orbit_radius_km = self.earth_radius_km + altitude_km
        self.mu = 398600.4418  # Earth's standard gravitational parameter [km^3/s^2]

        # Compute orbital period
        self.orbital_period_s = 2 * np.pi * np.sqrt(
            (self.orbit_radius_km ** 3) / self.mu
        )
        self.orbital_period_min = self.orbital_period_s / 60.0

        # Power model parameters
        self.eclipse_fraction = self._compute_eclipse_fraction()
        self.solar_power_generation_w = 200.0
        self.battery_discharge_rate_w = 80.0
        self.battery_charge_efficiency = 0.95

    def _compute_eclipse_fraction(self) -> float:
        """Compute fraction of orbit in eclipse (simplified)"""
        # For circular orbit: eclipse angle = 2 * arccos(R_earth / R_orbit)
        cos_half_eclipse = self.earth_radius_km / self.orbit_radius_km
        eclipse_angle_rad = 2 * np.arccos(cos_half_eclipse)
        eclipse_fraction = eclipse_angle_rad / (2 * np.pi)
        return eclipse_fraction

    def propagate(self, state: SatelliteState, dt_s: float,
                  is_sunlit: bool) -> SatelliteState:
        """
        Propagate satellite state forward in time

        Args:
            state: Current satellite state
            dt_s: Time step in seconds
            is_sunlit: Whether satellite is currently sunlit (for power generation)

        Returns:
            Updated satellite state
        """
        # Update orbital position (simplified - just increment epoch)
        new_state = state
        new_state.epoch += dt_s

        # Longitude drifts due to Earth's rotation and precession (simplified)
        new_state.longitude_deg = (new_state.longitude_deg +
                                    (360.0 / self.orbital_period_s) * dt_s) % 360.0

        # Latitude oscillates with orbital motion (simplified)
        phase = (new_state.epoch % self.orbital_period_s) / self.orbital_period_s
        new_state.latitude_deg = self.inclination_deg * np.sin(2 * np.pi * phase)

        # Update power
        new_state = self._update_power(new_state, dt_s, is_sunlit)

        return new_state

    def _update_power(self, state: SatelliteState, dt_s: float,
                     is_sunlit: bool) -> SatelliteState:
        """Update battery state of charge"""
        dt_h = dt_s / 3600.0

        if is_sunlit:
            # Generate power from solar panels
            power_balance_w = self.solar_power_generation_w - self.battery_discharge_rate_w
            energy_change_wh = power_balance_w * dt_h * self.battery_charge_efficiency
        else:
            # In eclipse - only discharge from battery
            energy_change_wh = -self.battery_discharge_rate_w * dt_h

        # Update battery
        new_energy_wh = (state.battery_soc * state.battery_capacity_wh) + energy_change_wh
        new_soc = np.clip(new_energy_wh / state.battery_capacity_wh, 0.0, 1.0)

        state.battery_soc = new_soc
        state.power_generation_w = self.solar_power_generation_w if is_sunlit else 0.0

        return state

    def slew_to_target(self, current_attitude: Tuple[float, float, float],
                       target_latitude: float, target_longitude: float,
                       dt_s: float) -> Tuple[Tuple[float, float, float], bool]:
        """
        Slew satellite attitude towards target

        Simplified model: directly calculate pointing angle needed

        Args:
            current_attitude: Current (pitch, roll, yaw) in degrees
            target_latitude: Target latitude in degrees
            target_longitude: Target longitude in degrees
            dt_s: Time step in seconds

        Returns:
            (new_attitude, is_pointing_at_target)
        """
        pitch, roll, yaw = current_attitude

        # Simplified: compute required pitch/roll to point at target
        # In reality this would be 3D geometry, but for RL we approximate
        lat_error = target_latitude - 0.0  # Current lat is ~0 at equator
        lon_error = target_longitude - 0.0

        required_pitch = np.clip(lat_error, -70, 70)  # Limit to achievable angles
        required_roll = np.clip(lon_error, -45, 45)

        # Compute max slew distance in this time step
        max_slew_deg = self.max_slew_rate_deg_s * dt_s

        # Move towards target with rate constraint
        new_pitch = pitch + np.clip(required_pitch - pitch, -max_slew_deg, max_slew_deg)
        new_roll = roll + np.clip(required_roll - roll, -max_slew_deg, max_slew_deg)

        # Check if pointing at target (within tolerance)
        pitch_error = abs(required_pitch - new_pitch)
        roll_error = abs(required_roll - new_roll)
        is_pointing = pitch_error < 2.0 and roll_error < 2.0

        return (new_pitch, new_roll, yaw), is_pointing

    def observe_target(self, state: SatelliteState, is_pointing: bool,
                      duration_s: float, data_rate_mbps: float = 10.0) -> float:
        """
        Generate observational data while pointing at target

        Args:
            state: Current satellite state
            is_pointing: Whether satellite is properly pointed at target
            duration_s: Observation duration in seconds
            data_rate_mbps: Data generation rate in Mbps

        Returns:
            Data generated in GB
        """
        if not is_pointing:
            return 0.0

        # Compute data volume
        duration_s_clamped = min(duration_s, 300)  # Max 5 min observations
        data_gb = (data_rate_mbps / 1000.0) * (duration_s_clamped / 3600.0)

        # Add some noise
        data_gb *= np.random.uniform(0.9, 1.1)

        return data_gb
