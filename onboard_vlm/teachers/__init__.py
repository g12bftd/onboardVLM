"""Teachers. Importing this package registers them."""

from onboard_vlm.teachers.feature_cache import TeacherFeatureCache, TeacherFeatureCacheWriter
from onboard_vlm.teachers.gemma4 import EncoderFreeTeacherError, Gemma4Teacher, pool_patch_states

__all__ = [
    "EncoderFreeTeacherError",
    "Gemma4Teacher",
    "TeacherFeatureCache",
    "TeacherFeatureCacheWriter",
    "pool_patch_states",
]
