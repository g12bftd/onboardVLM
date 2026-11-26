"""Student encoders. Importing this package registers all of them."""

from onboard_vlm.students.gemma4_shrink import Gemma4ShrinkStudent
from onboard_vlm.students.gemma4_vit import Gemma4ViTStudent
from onboard_vlm.students.multiplexed_vit import MultiplexedViT
from onboard_vlm.students.timm_spatial import TimmSpatialStudent
from onboard_vlm.students.vit import CoarsePatchViT, CompactViT, TokenViT

__all__ = ["CoarsePatchViT", "CompactViT", "Gemma4ShrinkStudent", "Gemma4ViTStudent", "MultiplexedViT", "TimmSpatialStudent", "TokenViT"]
