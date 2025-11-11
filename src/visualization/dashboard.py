"""
Streamlit Dashboard for AEOS-RL Visualization

Interactive dashboard for monitoring satellite scheduling and RL training.
Centerpiece: 3D orbital visualization with modular side controls.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualization.orbital_3d import OrbitVisualizer3D
from src.visualization.gantt_chart import GanttChartBuilder, ActivityType
from src.visualization.metrics import MetricsVisualizer


def setup_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title="AEOS-RL Satellite Scheduler",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        .main { padding: 0px; }
        .metrics-row { display: flex; gap: 20px; }
        </style>
    """, unsafe_allow_html=True)


def show_header():
    """Display dashboard header"""
    st.title("🛰️ AEOS-RL: Satellite Scheduling Dashboard")
    st.markdown("*Agile Earth Observation Satellite with Reinforcement Learning*")


def show_3d_visualization():
    """Display 3D orbital visualization (centerpiece)"""
    st.header("Orbital Visualization")

    visualizer = OrbitVisualizer3D()

    # Simulated satellite position
    satellite_pos = {
        "name": "AEOS-1",
        "lat": np.sin(np.random.rand() * 2 * np.pi) * 45,
        "lon": np.random.rand() * 360 - 180,
        "alt": 500,
        "color": "rgb(255, 100, 100)",
    }

    # Ground stations
    ground_stations = [
        {"name": "Fairbanks", "lat": 64.8, "lon": -147.7, "color": "rgb(100, 200, 100)"},
        {"name": "Santiago", "lat": -33.4, "lon": -70.4, "color": "rgb(100, 200, 100)"},
        {"name": "Singapore", "lat": 1.4, "lon": 103.8, "color": "rgb(100, 200, 100)"},
    ]

    # Sample observation targets
    observation_targets = [
        {"lat": 40.7, "lon": -74.0, "priority": 0.9, "name": "New York"},
        {"lat": 51.5, "lon": -0.1, "priority": 0.7, "name": "London"},
        {"lat": 35.7, "lon": 139.7, "priority": 0.8, "name": "Tokyo"},
        {"lat": -33.9, "lon": 151.2, "priority": 0.6, "name": "Sydney"},
    ]

    # Orbit trace
    orbit_traces = [
        {"altitude_km": 500, "inclination_deg": 97.5, "color": "rgb(100, 150, 255)", "name": "Current Orbit"}
    ]

    fig = visualizer.create_figure(
        satellite_positions=[satellite_pos],
        ground_stations=ground_stations,
        observation_targets=observation_targets,
        orbit_traces=orbit_traces,
        title="Real-time Satellite Trajectory",
    )

    st.plotly_chart(fig, use_container_width=True, height=600)


def show_metrics_panel():
    """Display metrics in sidebar"""
    st.sidebar.header("📊 Real-time Metrics")

    # Simulate metrics
    metrics = {
        "Battery SoC": 0.75,
        "Storage Used": 0.35,
        "Tasks Completed": 5,
        "Tasks Pending": 12,
        "Episode Progress": 0.42,
    }

    for metric_name, value in metrics.items():
        if isinstance(value, float):
            st.sidebar.metric(metric_name, f"{value*100:.1f}%" if value <= 1 else f"{value:.2f}")
        else:
            st.sidebar.metric(metric_name, value)


def show_schedule_panel():
    """Display task scheduling Gantt chart"""
    st.header("Task Schedule")

    builder = GanttChartBuilder()

    # Add sample activities
    builder.add_observation_task(1, 0, 300, 0.8, "Target A")
    builder.add_observation_task(2, 400, 300, 0.6, "Target B")
    builder.add_downlink_activity(700, 200, 2.5, "Fairbanks")
    builder.add_charging_activity(900, 500)
    builder.add_observation_task(3, 1400, 300, 0.9, "Target C")

    fig_gantt = builder.create_gantt_figure("Satellite Activity Timeline")
    st.plotly_chart(fig_gantt, use_container_width=True)

    # Resource timeline
    time_points = np.arange(0, 2000, 100)
    battery = 0.9 - 0.3 * (time_points / 2000) + 0.1 * np.sin(time_points / 300)
    battery = np.clip(battery, 0, 1)
    storage = 0.2 + 0.4 * (time_points / 2000) - 0.1 * np.cos(time_points / 400)
    storage = np.clip(storage, 0, 1)

    col1, col2 = st.columns(2)

    with col1:
        fig_battery = MetricsVisualizer.create_time_series(
            time_points,
            battery * 100,
            title="Battery State of Charge",
            ylabel="SoC %",
        )
        st.plotly_chart(fig_battery, use_container_width=True)

    with col2:
        fig_storage = MetricsVisualizer.create_time_series(
            time_points,
            storage * 10,  # In GB
            title="Data Storage",
            ylabel="Storage (GB)",
            color="rgb(100, 150, 255)",
        )
        st.plotly_chart(fig_storage, use_container_width=True)


def show_training_panel():
    """Display training metrics"""
    st.header("Training Performance")

    # Simulate training data
    episodes = np.arange(0, 100, 5)
    rewards = 10 * episodes + 50 * np.sin(episodes / 20) + np.random.randn(len(episodes)) * 20

    col1, col2 = st.columns(2)

    with col1:
        fig_training = MetricsVisualizer.create_training_curves(
            rewards.tolist(),
            episode_lengths=[90] * len(episodes),
            window_size=5,
        )
        st.plotly_chart(fig_training, use_container_width=True)

    with col2:
        completed = 42
        pending = 18
        failed = 3
        fig_completion = MetricsVisualizer.create_task_completion_chart(
            completed, pending, failed
        )
        st.plotly_chart(fig_completion, use_container_width=True)


def show_algorithm_comparison():
    """Show algorithm comparison"""
    st.header("Algorithm Comparison")

    algorithms = ["Random", "Greedy", "EDF", "PPO"]
    metrics_data = {
        "Random": [50, 50, 50],
        "Greedy": [150, 150, 150],
        "EDF": [160, 160, 160],
        "PPO": [200, 200, 200],
    }

    fig_comparison = MetricsVisualizer.create_algorithm_comparison(
        algorithms,
        metrics_data,
        metric_name="Average Reward per Episode",
    )
    st.plotly_chart(fig_comparison, use_container_width=True)


def show_controls():
    """Show control panel in sidebar"""
    st.sidebar.header("⚙️ Controls")

    st.sidebar.subheader("Simulation Control")
    simulation_speed = st.sidebar.slider("Simulation Speed (x)", 0.1, 5.0, 1.0)
    is_running = st.sidebar.checkbox("Running", value=True)

    st.sidebar.subheader("Visualization Options")
    show_orbit = st.sidebar.checkbox("Show Orbit", value=True)
    show_ground_tracks = st.sidebar.checkbox("Show Ground Tracks", value=True)
    show_targets = st.sidebar.checkbox("Show Targets", value=True)

    st.sidebar.subheader("Display Mode")
    display_mode = st.sidebar.radio(
        "Select View",
        ["Orbital", "Schedule", "Training", "Comparison"],
    )

    return display_mode, is_running, simulation_speed


def main():
    """Main dashboard function"""
    setup_page()
    show_header()

    # Sidebar controls
    display_mode, is_running, sim_speed = show_controls()
    show_metrics_panel()

    # Main content based on selection
    if display_mode == "Orbital":
        show_3d_visualization()

    elif display_mode == "Schedule":
        show_schedule_panel()

    elif display_mode == "Training":
        show_training_panel()

    elif display_mode == "Comparison":
        show_algorithm_comparison()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **AEOS-RL Dashboard** |
        Powered by Streamlit, Plotly, and Stable-Baselines3 |
        [GitHub](https://github.com) |
        [Documentation](#)
        """
    )


if __name__ == "__main__":
    main()
