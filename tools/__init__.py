"""
Tools package for adversarial robustness, privacy analysis, and explainability.
"""

from .attacks import *
from .privacy import *
from .xai import *
from .eval_metrics import *

__all__ = [
    'fgsm_attack',
    'pgd_attack',
    'cw_attack',
    'MembershipInferenceAttack',
    'train_with_dp',
    'GradCAM',
    'OcclusionSensitivity',
    'IntegratedGradients',
    'extract_features',
    'compute_tsne',
    'EvaluationMetrics'
]
