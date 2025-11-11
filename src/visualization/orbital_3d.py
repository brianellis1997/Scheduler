"""
3D Orbital Visualization Module

Creates interactive 3D visualizations of satellite orbits using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional, Dict, Any


class OrbitVisualizer3D:
    """Creates 3D orbital visualization with satellite trajectory and ground stations"""

    def __init__(self, earth_radius_km: float = 6371.0):
        """
        Initialize orbit visualizer

        Args:
            earth_radius_km: Earth radius in kilometers
        """
        self.earth_radius_km = earth_radius_km

    def create_earth_sphere(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create Earth sphere coordinates for 3D plot

        Args:
            n_points: Resolution of sphere mesh

        Returns:
            (x, y, z) arrays for Earth surface
        """
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        x = self.earth_radius_km * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius_km * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))

        return x, y, z

    def latlon_to_cartesian(self, latitude_deg: float, longitude_deg: float,
                            altitude_km: float = 0) -> Tuple[float, float, float]:
        """
        Convert lat/lon/altitude to Cartesian coordinates

        Args:
            latitude_deg: Latitude in degrees
            longitude_deg: Longitude in degrees
            altitude_km: Altitude above Earth surface in km

        Returns:
            (x, y, z) Cartesian coordinates in km
        """
        lat_rad = np.radians(latitude_deg)
        lon_rad = np.radians(longitude_deg)
        r = self.earth_radius_km + altitude_km

        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)

        return x, y, z

    def create_orbit_trace(self, altitude_km: float, inclination_deg: float,
                          num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create orbital trajectory for visualization

        Args:
            altitude_km: Orbital altitude
            inclination_deg: Orbital inclination
            num_points: Number of points along orbit

        Returns:
            (x, y, z) arrays for orbit trajectory
        """
        # Create circular orbit trajectory
        angle = np.linspace(0, 2 * np.pi, num_points)
        r = self.earth_radius_km + altitude_km

        inclination_rad = np.radians(inclination_deg)

        # Orbit in orbital plane
        x_orbit = r * np.cos(angle)
        y_orbit = r * np.sin(angle) * np.cos(inclination_rad)
        z_orbit = r * np.sin(angle) * np.sin(inclination_rad)

        return x_orbit, y_orbit, z_orbit

    def create_ground_track(self, latitude_deg: float, inclination_deg: float,
                           num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create ground track (projection of orbit on Earth surface)

        Args:
            latitude_deg: Current latitude
            inclination_deg: Orbital inclination
            num_points: Number of points

        Returns:
            (lat_track, lon_track) arrays for ground track
        """
        # Simplified ground track as function of orbit angle
        angle = np.linspace(0, 2 * np.pi, num_points)

        # Inclination determines max latitude
        max_lat = inclination_deg
        lat_track = max_lat * np.sin(angle)

        # Longitude drifts due to Earth rotation and orbital motion
        lon_track = np.degrees(angle) - (360 * np.linspace(0, 1, num_points))

        return lat_track, lon_track

    def create_figure(self, satellite_positions: List[Dict[str, Any]],
                     ground_stations: Optional[List[Dict[str, Any]]] = None,
                     observation_targets: Optional[List[Dict[str, Any]]] = None,
                     orbit_traces: Optional[List[Dict[str, Any]]] = None,
                     title: str = "AEOS Orbital Visualization") -> go.Figure:
        """
        Create complete 3D visualization figure

        Args:
            satellite_positions: List of dicts with 'lat', 'lon', 'alt', 'name'
            ground_stations: List of dicts with 'lat', 'lon', 'name'
            observation_targets: List of dicts with 'lat', 'lon', 'priority'
            orbit_traces: List of dicts with 'altitude_km', 'inclination_deg', 'color'
            title: Figure title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Add Earth sphere
        x_earth, y_earth, z_earth = self.create_earth_sphere()
        fig.add_trace(go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            colorscale="Blues",
            showscale=False,
            opacity=0.8,
            name="Earth",
            hovertemplate="<b>Earth</b><extra></extra>",
        ))

        # Add orbit traces
        if orbit_traces:
            for orbit in orbit_traces:
                x_orbit, y_orbit, z_orbit = self.create_orbit_trace(
                    orbit.get("altitude_km", 500),
                    orbit.get("inclination_deg", 97.5),
                )
                fig.add_trace(go.Scatter3d(
                    x=x_orbit,
                    y=y_orbit,
                    z=z_orbit,
                    mode="lines",
                    name=orbit.get("name", "Orbit"),
                    line=dict(
                        color=orbit.get("color", "rgb(100, 150, 255)"),
                        width=3,
                    ),
                    hovertemplate="<b>Orbital Path</b><extra></extra>",
                ))

        # Add satellites
        if satellite_positions:
            for sat in satellite_positions:
                x, y, z = self.latlon_to_cartesian(
                    sat.get("lat", 0),
                    sat.get("lon", 0),
                    sat.get("alt", 500),
                )
                fig.add_trace(go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode="markers+text",
                    name=sat.get("name", "Satellite"),
                    marker=dict(
                        size=12,
                        color=sat.get("color", "rgb(255, 100, 100)"),
                        symbol="diamond",
                    ),
                    text=sat.get("name", "Sat"),
                    textposition="top center",
                    hovertemplate=f"<b>{sat.get('name', 'Satellite')}</b><br>" +
                                "Lat: %{customdata[0]:.2f}°<br>" +
                                "Lon: %{customdata[1]:.2f}°<br>" +
                                "Alt: {:.0f} km<extra></extra>".format(sat.get("alt", 500)),
                    customdata=[[sat.get("lat", 0), sat.get("lon", 0)]],
                ))

        # Add ground stations
        if ground_stations:
            for station in ground_stations:
                x, y, z = self.latlon_to_cartesian(
                    station.get("lat", 0),
                    station.get("lon", 0),
                    0,
                )
                fig.add_trace(go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode="markers+text",
                    name=station.get("name", "Station"),
                    marker=dict(
                        size=10,
                        color=station.get("color", "rgb(100, 200, 100)"),
                        symbol="square",
                    ),
                    text=station.get("name", "GS"),
                    textposition="bottom center",
                    hovertemplate=f"<b>{station.get('name', 'Ground Station')}</b><extra></extra>",
                ))

        # Add observation targets
        if observation_targets:
            for target in observation_targets:
                x, y, z = self.latlon_to_cartesian(
                    target.get("lat", 0),
                    target.get("lon", 0),
                    0,
                )
                priority = target.get("priority", 0.5)
                color_intensity = int(255 * priority)
                color = f"rgb(255, {color_intensity}, 0)"

                fig.add_trace(go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode="markers",
                    name=f"Target (P:{priority:.1f})",
                    marker=dict(
                        size=6,
                        color=color,
                        symbol="circle",
                        opacity=0.7,
                    ),
                    hovertemplate="<b>Observation Target</b><br>" +
                                f"Priority: {priority:.2f}<extra></extra>",
                ))

        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
                zaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
                aspectmode="data",
                camera=dict(
                    eye=dict(x=2, y=2, z=1.5),
                ),
            ),
            width=1000,
            height=800,
            hovermode="closest",
            showlegend=True,
            legend=dict(
                x=0.0,
                y=1.0,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
            ),
        )

        return fig
