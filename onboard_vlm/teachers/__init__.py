"""Teachers. Importing this package registers them."""

from onboard_vlm.teachers.feature_cache import TeacherFeatureCache, TeacherFeatureCacheWriter
from onboard_vlm.teachers.gemma4 import (
    SUPPORTED_TEACHERS,
    EncoderFreeTeacherError,
    Gemma4Teacher,
    UnsupportedTeacherError,
    load_gemma4_vlm,
    pool_patch_states,
    resolve_revision,
)

__all__ = [
    "SUPPORTED_TEACHERS",
    "EncoderFreeTeacherError",
    "Gemma4Teacher",
    "TeacherFeatureCache",
    "TeacherFeatureCacheWriter",
    "UnsupportedTeacherError",
    "load_gemma4_vlm",
    "pool_patch_states",
    "resolve_revision",
]
