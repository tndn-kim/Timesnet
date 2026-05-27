"""
Timesnet_CIC
============
CIC-UNSW-NB15 침입 탐지 분류용 TimesNet 패키지.

빠른 시작:
    from Timesnet_CIC.Timesnet import run_timesnet
    from Timesnet_CIC.visualize import visualize_all

    result    = run_timesnet(X, y, config={...}, class_names=[...])
    viz_paths = visualize_all(result, y, class_names=[...], save_dir="./output")
"""

from .Timesnet import Model, TimesNetConfig, run_timesnet, compute_metrics
from .visualize import visualize_all

__all__ = [
    "Model",
    "TimesNetConfig",
    "run_timesnet",
    "compute_metrics",
    "visualize_all",
]
