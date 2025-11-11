"""
Gantt Chart Visualization for Task Scheduling

Visualizes satellite task schedule and resource utilization timeline.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from enum import Enum


class ActivityType(Enum):
    """Types of satellite activities"""

    OBSERVATION = "Observation"
    DOWNLINK = "Downlink"
    CHARGING = "Charging"
    IDLE = "Idle"
    MAINTENANCE = "Maintenance"


class GanttChartBuilder:
    """Builds interactive Gantt charts for satellite scheduling"""

    # Color scheme for different activity types
    COLORS = {
        ActivityType.OBSERVATION: "#FF6B6B",
        ActivityType.DOWNLINK: "#4ECDC4",
        ActivityType.CHARGING: "#FFE66D",
        ActivityType.IDLE: "#95E1D3",
        ActivityType.MAINTENANCE: "#C7CEEA",
    }

    def __init__(self):
        """Initialize Gantt chart builder"""
        self.activities = []

    def add_observation_task(self, task_id: int, start_time_s: float,
                            duration_s: float, priority: float,
                            target_name: str = "Task") -> None:
        """
        Add observation activity

        Args:
            task_id: Unique task identifier
            start_time_s: Start time in seconds
            duration_s: Duration in seconds
            priority: Priority level [0, 1]
            target_name: Name of observation target
        """
        self.activities.append({
            "Task": target_name,
            "Type": ActivityType.OBSERVATION.value,
            "Start": start_time_s,
            "Finish": start_time_s + duration_s,
            "Duration": duration_s,
            "Priority": priority,
            "Resource": "Satellite",
            "Description": f"Observe {target_name} (Priority: {priority:.2f})",
        })

    def add_downlink_activity(self, start_time_s: float, duration_s: float,
                             data_size_gb: float, station_name: str = "GS") -> None:
        """Add data downlink activity"""
        self.activities.append({
            "Task": f"Downlink to {station_name}",
            "Type": ActivityType.DOWNLINK.value,
            "Start": start_time_s,
            "Finish": start_time_s + duration_s,
            "Duration": duration_s,
            "Priority": 0.5,
            "Resource": "Satellite",
            "Description": f"Downlink {data_size_gb:.2f} GB",
        })

    def add_charging_activity(self, start_time_s: float, duration_s: float) -> None:
        """Add battery charging activity"""
        self.activities.append({
            "Task": "Charging",
            "Type": ActivityType.CHARGING.value,
            "Start": start_time_s,
            "Finish": start_time_s + duration_s,
            "Duration": duration_s,
            "Priority": 0.3,
            "Resource": "Satellite",
            "Description": "Solar charging",
        })

    def create_gantt_figure(self, title: str = "Satellite Schedule") -> go.Figure:
        """
        Create Gantt chart figure

        Args:
            title: Chart title

        Returns:
            Plotly Figure object
        """
        if not self.activities:
            # Return empty figure if no activities
            return go.Figure().add_annotation(
                text="No activities to display",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
            )

        df = pd.DataFrame(self.activities)

        # Map activity types to colors
        df["Color"] = df["Type"].map(self.COLORS)

        # Create Gantt chart
        fig = px.timeline(
            df,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Type",
            color_discrete_map=self.COLORS,
            hover_name="Description",
            hover_data={"Start": False, "Finish": False, "Duration": ":.1f",
                       "Type": False, "Priority": ":.2f"},
            title=title,
            labels={"Start": "Time (s)", "Finish": "Time (s)"},
        )

        # Customize layout
        fig.update_layout(
            xaxis_type="linear",
            height=400,
            hovermode="closest",
            showlegend=True,
        )

        # Format x-axis as time
        fig.update_xaxes(
            title_text="Time (seconds)",
            tickformat=".0f",
        )

        return fig

    def create_resource_timeline(self, battery_data: List[Dict[str, Any]],
                                storage_data: List[Dict[str, Any]],
                                title: str = "Resource Utilization") -> go.Figure:
        """
        Create timeline of resource usage (battery, storage)

        Args:
            battery_data: List of dicts with 'time_s' and 'soc'
            storage_data: List of dicts with 'time_s' and 'gb'
            title: Chart title

        Returns:
            Plotly Figure with dual y-axes
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Battery State of Charge", "Data Storage"),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        )

        # Battery SoC
        if battery_data:
            battery_df = pd.DataFrame(battery_data)
            fig.add_trace(
                go.Scatter(
                    x=battery_df["time_s"],
                    y=battery_df["soc"] * 100,
                    mode="lines",
                    name="Battery SoC",
                    line=dict(color="rgb(255, 150, 100)", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(255, 150, 100, 0.3)",
                ),
                row=1, col=1,
            )

        # Storage
        if storage_data:
            storage_df = pd.DataFrame(storage_data)
            fig.add_trace(
                go.Scatter(
                    x=storage_df["time_s"],
                    y=storage_df["gb"],
                    mode="lines",
                    name="Data Storage",
                    line=dict(color="rgb(100, 150, 255)", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(100, 150, 255, 0.3)",
                ),
                row=2, col=1,
            )

        # Update axes
        fig.update_yaxes(title_text="Battery %", row=1, col=1, range=[0, 110])
        fig.update_yaxes(title_text="Storage (GB)", row=2, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)

        fig.update_layout(
            title_text=title,
            height=600,
            hovermode="x unified",
        )

        return fig


def make_subplots(*args, **kwargs):
    """Helper to import make_subplots"""
    from plotly.subplots import make_subplots as _make_subplots
    return _make_subplots(*args, **kwargs)
