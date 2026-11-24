"""NYU Depth v2 (official labelled subset) -> ``<out>/{train,test}/<idx>.png`` + ``<idx>_depth.npy``.

    curl -L -o $RAW/nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    curl -L -o $RAW/splits.mat http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
    python make_datasets/eval_sets/make_nyuv2.py --mat $RAW/nyu_depth_v2_labeled.mat --splits $RAW/splits.mat --out $EVAL_ROOT/nyuv2

The .mat is MATLAB v7.3 (HDF5): ``images`` [N, 3, W, H] uint8, ``depths`` [N, W, H] metres;
``splits.mat`` holds the standard 1-based ``trainNdxs`` (795) / ``testNdxs`` (654).
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mat", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import h5py
    from PIL import Image
    from scipy.io import loadmat

    sp = loadmat(args.splits)
    with h5py.File(args.mat, "r") as f:
        images, depths = f["images"], f["depths"]
        for split, key in (("train", "trainNdxs"), ("test", "testNdxs")):
            d = os.path.join(args.out, split)
            os.makedirs(d, exist_ok=True)
            for idx in sp[key].ravel().astype(int) - 1:
                rgb = np.transpose(images[idx], (2, 1, 0))  # -> H, W, 3
                Image.fromarray(rgb).save(os.path.join(d, f"{idx:04d}.png"))
                np.save(os.path.join(d, f"{idx:04d}_depth.npy"), np.transpose(depths[idx], (1, 0)).astype(np.float32))
            print(f"{split}: {len(sp[key])} images -> {d}")


if __name__ == "__main__":
    main()
