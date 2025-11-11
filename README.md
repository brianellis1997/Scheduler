# AEOS-RL: Agile Earth Observation Satellite Reinforcement Learning Scheduler

A cutting-edge reinforcement learning system for scheduling Earth observation satellite tasks using Proximal Policy Optimization (PPO). This project demonstrates advanced ML capabilities including RL model training, satellite dynamics simulation, and interactive visualization—directly aligned with modern space industry ML engineering.

## Overview

**AEOS-RL** solves the complex problem of scheduling observation tasks for agile Earth observation satellites while respecting operational constraints:
- Ground station visibility windows for data downlink
- Onboard power and battery constraints
- Data storage limitations
- Satellite slew angle constraints
- Task priorities and time windows

The system uses a trained PPO agent to learn optimal scheduling policies that maximize scientific value while maintaining operational compliance.

## Key Features

### Reinforcement Learning
- **Algorithm**: Proximal Policy Optimization (PPO) via Stable-Baselines3
- **Environment**: Custom Gymnasium-compatible AEOS environment with realistic dynamics
- **Training**: Full end-to-end training pipeline with TensorBoard monitoring
- **Baselines**: Comparison against greedy, random, and earliest-deadline-first heuristics

### Simulation & Dynamics
- Orbital mechanics using Skyfield (accurate TLE propagation)
- Realistic satellite dynamics (power generation, battery discharge, memory constraints)
- Ground station contact prediction and visibility windows
- Time-dependent task generation with priorities

### Visualization
- **3D Orbital Visualization** (centerpiece): Interactive Earth model with satellite orbit, ground track, and targets
- **Scheduling Gantt Charts**: Task timeline with color-coded activities
- **Resource Monitoring**: Real-time battery and memory utilization
- **Performance Metrics**: Task completion rates, latency analysis, efficiency metrics
- **Training Dashboard**: Interactive Streamlit app with modular controls

## Quick Start

### Installation

```bash
# Using conda
conda env create -f environment.yml
conda activate aeos-scheduler

# Or using pip
pip install -r requirements.txt
```

### Train a PPO Agent

```bash
python src/models/ppo_trainer.py --config configs/training.yaml
```

### Run Interactive Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

## Project Structure

```
Scheduler/
├── src/                     # Source code
│   ├── environment/         # Custom Gymnasium environment
│   ├── models/              # RL agent and training
│   ├── simulation/          # Satellite dynamics and orbital mechanics
│   ├── visualization/       # Plotting and dashboard
│   └── utils/               # Utility functions
├── notebooks/               # Jupyter demos
├── configs/                 # Configuration files
├── data/                    # Input data
├── logs/                    # TensorBoard logs
├── models/                  # Trained checkpoints
└── requirements.txt         # Dependencies
```

## Technologies

- **RL Framework**: PyTorch + Stable-Baselines3
- **Environment API**: Gymnasium
- **Orbital Mechanics**: Skyfield
- **Visualization**: Plotly + Streamlit
- **Monitoring**: TensorBoard

## Status

**Current Phase**: Phase 1 - Foundation & Setup
**Last Updated**: 2025-11-10
