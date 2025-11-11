"""
Performance Metrics Visualization

Creates plots for training performance, task completion, and algorithm comparison.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional


class MetricsVisualizer:
    """Visualizes training and performance metrics"""

    @staticmethod
    def create_training_curves(episode_rewards: List[float],
                              episode_lengths: List[int],
                              window_size: int = 10,
                              title: str = "Training Progress") -> go.Figure:
        """
        Create training reward curves with moving average

        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            window_size: Moving average window size
            title: Figure title

        Returns:
            Plotly Figure
        """
        df = pd.DataFrame({
            "Episode": np.arange(len(episode_rewards)),
            "Reward": episode_rewards,
            "Length": episode_lengths,
        })

        # Calculate moving averages
        df["Reward MA"] = df["Reward"].rolling(window=window_size, center=True).mean()

        fig = go.Figure()

        # Raw rewards (semi-transparent)
        fig.add_trace(go.Scatter(
            x=df["Episode"],
            y=df["Reward"],
            mode="markers",
            name="Episode Reward",
            marker=dict(
                size=4,
                color="rgba(100, 150, 255, 0.3)",
            ),
            hovertemplate="<b>Episode %{x}</b><br>Reward: %{y:.2f}<extra></extra>",
        ))

        # Moving average (prominent)
        fig.add_trace(go.Scatter(
            x=df["Episode"],
            y=df["Reward MA"],
            mode="lines",
            name=f"Moving Average ({window_size})",
            line=dict(
                color="rgb(0, 100, 200)",
                width=3,
            ),
            hovertemplate="<b>Episode %{x}</b><br>Avg Reward: %{y:.2f}<extra></extra>",
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Episode",
            yaxis_title="Cumulative Reward",
            hovermode="x unified",
            height=400,
        )

        return fig

    @staticmethod
    def create_task_completion_chart(completed: int, pending: int, failed: int,
                                    title: str = "Task Completion Status") -> go.Figure:
        """
        Create task completion pie chart

        Args:
            completed: Number of completed tasks
            pending: Number of pending tasks
            failed: Number of failed tasks
            title: Chart title

        Returns:
            Plotly Figure
        """
        labels = ["Completed", "Pending", "Failed"]
        values = [completed, pending, failed]
        colors = ["rgb(76, 205, 196)", "rgb(255, 230, 109)", "rgb(255, 107, 107)"]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hovertemplate="<b>%{label}</b><br>Tasks: %{value}<br>%{percent}<extra></extra>",
        )])

        fig.update_layout(
            title=title,
            height=400,
        )

        return fig

    @staticmethod
    def create_algorithm_comparison(algorithms: List[str],
                                   metrics: Dict[str, List[float]],
                                   metric_name: str = "Average Reward",
                                   title: str = "Algorithm Comparison") -> go.Figure:
        """
        Create bar chart comparing multiple algorithms

        Args:
            algorithms: List of algorithm names
            metrics: Dict of algorithm_name -> metric_values
            metric_name: Name of metric being compared
            title: Chart title

        Returns:
            Plotly Figure
        """
        data = []
        for algo in algorithms:
            values = metrics.get(algo, [0])
            mean_val = np.mean(values)
            std_val = np.std(values)
            data.append({
                "Algorithm": algo,
                "Mean": mean_val,
                "Std": std_val,
            })

        df = pd.DataFrame(data)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df["Algorithm"],
            y=df["Mean"],
            error_y=dict(type="data", array=df["Std"]),
            name=metric_name,
            marker=dict(
                color="rgb(100, 150, 255)",
            ),
            hovertemplate="<b>%{x}</b><br>" + metric_name + ": %{y:.2f}<extra></extra>",
        ))

        fig.update_layout(
            title=title,
            yaxis_title=metric_name,
            xaxis_title="Algorithm",
            height=400,
        )

        return fig

    @staticmethod
    def create_heatmap(data: np.ndarray, x_labels: List[str],
                      y_labels: List[str], title: str = "Heatmap") -> go.Figure:
        """
        Create heatmap visualization

        Args:
            data: 2D numpy array of values
            x_labels: X-axis labels
            y_labels: Y-axis labels
            title: Figure title

        Returns:
            Plotly Figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            hovertemplate="<b>%{y} vs %{x}</b><br>Value: %{z:.3f}<extra></extra>",
        ))

        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title="",
            height=500,
        )

        return fig

    @staticmethod
    def create_histogram(data: List[float], bins: int = 30,
                        title: str = "Distribution",
                        xlabel: str = "Value") -> go.Figure:
        """
        Create histogram

        Args:
            data: List of values
            bins: Number of histogram bins
            title: Figure title
            xlabel: X-axis label

        Returns:
            Plotly Figure
        """
        fig = go.Figure(data=[go.Histogram(
            x=data,
            nbinsx=bins,
            marker=dict(color="rgb(100, 150, 255)"),
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        )])

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title="Frequency",
            height=400,
        )

        return fig

    @staticmethod
    def create_time_series(time_points: List[float], values: List[float],
                          title: str = "Time Series",
                          ylabel: str = "Value",
                          color: str = "rgb(100, 150, 255)") -> go.Figure:
        """
        Create time series line plot

        Args:
            time_points: Time values
            values: Data values
            title: Figure title
            ylabel: Y-axis label
            color: Line color

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_points,
            y=values,
            mode="lines",
            name="Value",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.2)"),
            hovertemplate="<b>Time: %{x:.0f}s</b><br>" + ylabel + ": %{y:.2f}<extra></extra>",
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title=ylabel,
            hovermode="x unified",
            height=400,
        )

        return fig

    @staticmethod
    def create_dashboard_summary(metrics_dict: Dict[str, Any],
                                title: str = "Performance Summary") -> go.Figure:
        """
        Create summary dashboard with key metrics

        Args:
            metrics_dict: Dictionary of metric_name -> value
            title: Dashboard title

        Returns:
            Plotly Figure with custom annotations
        """
        fig = go.Figure()

        # Add invisible trace (needed for proper rendering)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers"))

        # Add metric boxes as annotations
        y_position = 0.9
        for name, value in metrics_dict.items():
            if isinstance(value, float):
                text = f"<b>{name}</b><br>{value:.3f}"
            else:
                text = f"<b>{name}</b><br>{value}"

            fig.add_annotation(
                text=text,
                x=0.2,
                y=y_position,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(200, 220, 255, 0.7)",
                bordercolor="rgb(0, 100, 200)",
                borderwidth=1,
                borderpad=10,
            )

            y_position -= 0.15

        fig.update_layout(
            title=title,
            xaxis_visible=False,
            yaxis_visible=False,
            height=300 + (len(metrics_dict) * 30),
        )

        return fig
