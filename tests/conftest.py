import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "make_datasets", "lmdb"))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def tiny_cfg():
    from onboard_vlm.utils.tiny import tiny_gemma4_config

    return tiny_gemma4_config(vision_hidden=48, vision_layers=3)


@pytest.fixture(scope="session")
def tiny_teacher(tiny_cfg):
    from onboard_vlm.teachers.gemma4 import Gemma4Teacher

    return Gemma4Teacher(config=tiny_cfg, load="vision", dtype="float32")


@pytest.fixture(scope="session")
def tiny_processor():
    from onboard_vlm.utils.tiny import tiny_gemma4_processor

    return tiny_gemma4_processor()


def write_video(path: str, num_frames: int = 30, fps: int = 10, size: tuple[int, int] = (160, 96)) -> None:
    """Small synthetic MP4 whose frame i has brightness ~ 8*i (lets tests check which frame was read)."""
    import av
    import numpy as np

    with av.open(path, "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(num_frames):
            arr = np.full((size[1], size[0], 3), min(255, 8 * i), dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
