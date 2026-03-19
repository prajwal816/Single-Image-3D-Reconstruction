"""Explainability: SHAP analysis and feature importance for 3D reconstruction."""

from .shap_analysis import SHAPAnalyzer
from .feature_importance import GradientSaliency, OcclusionSensitivity
