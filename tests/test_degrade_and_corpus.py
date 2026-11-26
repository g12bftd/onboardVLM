"""Capture degradations (consistent teaching), fingerprints, corpus builders and decontamination."""

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import ROOT, write_video
from PIL import Image

from onboard_vlm import geometry as G
from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.data.corpus import ImageCorpusDataset
from onboard_vlm.data.degrade import DEFAULT_PROBS, apply_degradations, sample_degradations
from onboard_vlm.data.hashing import NearDupIndex, dhash, hamming
from onboard_vlm.interfaces import TransformParams

CORPUS = os.path.join(ROOT, "make_datasets", "corpus")
ALL_OPS = {k: 1.0 for k in DEFAULT_PROBS}


def _run(script, *args):
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, os.path.join(CORPUS, script), *args], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


@pytest.mark.parametrize("op", sorted(DEFAULT_PROBS))
def test_each_degradation_is_deterministic_and_shape_preserving(op):
    img = torch.randint(0, 256, (3, 96, 144), dtype=torch.uint8)
    ops = sample_degradations({op: 1.0}, random.Random(0), (96, 144))
    assert [o[0] for o in ops] == [op]
    a, b = apply_degradations(img, ops), apply_degradations(img, ops)
    assert torch.equal(a, b) and a.shape == img.shape and a.dtype == torch.uint8
    if op != "rotate" or ops[0][1][0] != 180.0:
        assert not torch.equal(a, img)


def test_degraded_transform_replays_through_json():
    img = torch.randint(0, 256, (3, 400, 600), dtype=torch.uint8)
    p = G.sample_transform((400, 600), 140, train=True, rng=random.Random(3), degradations=ALL_OPS)
    assert len(p.degradations) == len(ALL_OPS)
    p2 = TransformParams.from_dict(json.loads(json.dumps(p.to_dict())))
    assert p2 == p and torch.equal(G.apply_transform(img, p), G.apply_transform(img, p2))
    assert G.sample_transform((400, 600), 140, train=False, degradations=ALL_OPS).degradations == ()


def test_teacher_and_student_see_identical_degraded_pixels():
    samples = [{"sample_id": str(i), "image": torch.randint(0, 256, (3, 300 + 50 * i, 500), dtype=torch.uint8)} for i in range(3)]
    b = DualViewCollator([70, 140], train=True, degradations=ALL_OPS, seed=1)(samples)
    assert all(t.degradations for t in b.transforms)
    dense, mask, _ = G.unpatchify_batch(b.teacher_inputs.pixel_values, b.teacher_inputs.pixel_position_ids)
    assert torch.equal(dense, b.student_images[:, :, : dense.shape[2], : dense.shape[3]]) and torch.equal(mask, b.valid_token_mask)


def test_dhash_and_near_dup_index():
    img = Image.effect_mandelbrot((256, 192), (-2, -1.5, 1, 1.5), 50).convert("RGB")
    h = dhash(img)
    assert hamming(h, dhash(img.resize((128, 96)))) <= 4  # robust to resizing
    other = dhash(Image.effect_noise((256, 192), 60).convert("RGB"))
    idx = NearDupIndex(max_dist=4)
    idx.add(h)
    assert idx.contains(h ^ 0b1011) and not idx.contains(other) and len(idx) == 1


def _structured(seed: int, size=(320, 240)) -> Image.Image:
    """Smooth random image (passes the near-uniform filter, distinct dHash per seed)."""
    g = torch.Generator().manual_seed(seed)
    small = torch.randint(0, 256, (6, 8, 3), generator=g, dtype=torch.uint8).numpy()
    return Image.fromarray(small).resize(size, Image.BICUBIC)


def test_image_corpus_builder_filters_and_decontaminates(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    base = Image.effect_mandelbrot((320, 240), (-2, -1.5, 1, 1.5), 60).convert("RGB")
    base.save(src / "a.jpg")
    base.resize((300, 225)).save(src / "a_resized_dup.jpg")  # perceptual duplicate: removed by finalize, not here
    Image.new("RGB", (100, 80), (1, 2, 3)).save(src / "tiny.jpg")  # too small
    _structured(1, (2000, 240)).save(src / "panorama.jpg")  # aspect > 4
    Image.new("RGB", (320, 240), (90, 90, 90)).save(src / "blank.jpg")  # near-uniform
    bench = _structured(2)
    bench.save(src / "benchmark_copy.png")
    _structured(3).save(src / "0123abcd.jpg")  # an Open Images id used by a TextVQA val image
    bl = tmp_path / "bl.json"
    json.dump({"sha1": [], "dhash": [f"{dhash(bench):016x}"], "source_ids": {"openimages": ["0123abcd"]}, "video_ids": [], "video_spans": []}, open(bl, "w"))
    out = tmp_path / "corpus"
    _run("make_image_corpus.py", "--source", "folder", "--input_path", str(src), "--name", "toy", "--output_path", str(out),
         "--blocklist", str(bl), "--id_key", "openimages", "--shard_size", "1")
    stats = json.load(open(out / "stats_000.json"))
    assert stats == {"seen": 7, "written": 2, "decode_error": 0, "too_small": 1, "bad_aspect": 1, "near_uniform": 1, "benchmark_dup": 2}
    ds = ImageCorpusDataset(str(out))
    assert len(ds) == 2 and ds.sizes.tolist() == [[240, 320], [225, 300]]
    assert ds[0]["image"].shape == (3, 240, 320) and ds[0]["sample_id"] == "toy/a.jpg"
    with pytest.raises(FileNotFoundError, match="finalize"):
        ImageCorpusDataset(str(out), require_finalized=True)


def _two_source_corpus(tmp_path):
    raw = {k: tmp_path / "raw" / k for k in ("a", "b")}
    for d in raw.values():
        d.mkdir(parents=True)
    for i in range(30):
        _structured(10 + i).save(raw["a"] / f"{i:02d}.jpg", quality=90)
    for i in range(3):  # byte-identical copies in source b (exact cross-source duplicates)
        (raw["b"] / f"copy{i}.jpg").write_bytes((raw["a"] / f"{i:02d}.jpg").read_bytes())
    _structured(15).resize((300, 225)).save(raw["b"] / "resized.jpg")  # perceptual cross-source duplicate
    for i in range(20):
        _structured(100 + i).save(raw["b"] / f"new{i}.jpg")
    corpus = tmp_path / "corpus"
    for k, d in raw.items():
        _run("make_image_corpus.py", "--source", "folder", "--input_path", str(d), "--name", k, "--output_path", str(corpus / k), "--shard_size", "7")
    spec = tmp_path / "sources.yaml"
    spec.write_text("name: toy\nversion: 1\nsources:\n  - {dir: a, group: g, license: x}\n  - {dir: b, group: g, license: y}\n")
    return corpus, spec


def test_finalize_dedups_across_sources(tmp_path):
    corpus, spec = _two_source_corpus(tmp_path)
    _run("finalize_corpus.py", "--corpus_root", str(corpus), "--sources_yaml", str(spec), "--val_fraction", "0.2", "--test_fraction", "0.2")
    m = json.load(open(corpus / "corpus_manifest.json"))
    a, b = m["sources"]
    assert (a["built"], a["kept"]) == (30, 30)
    assert (b["built"], b["dup_exact_cross"], b["dup_dhash_cross"], b["kept"]) == (24, 3, 1, 20)
    assert m["total_kept"] == 50 and sum(m["splits"].values()) == 50 and (corpus / "CORPUS_CARD.md").exists()
    assert json.load(open(corpus / "b" / "dataset.json"))["license"] == "y"
    # the loader reproduces the split sizes the card reports, for the same definition
    split = {"val_fraction": 0.2, "test_fraction": 0.2, "split_seed": 0, "require_finalized": True}
    parts = {sp: ImageCorpusDataset(str(corpus / "*"), split=sp, **split) for sp in ("train", "val", "test")}
    assert {sp: len(d) for sp, d in parts.items()} == m["splits"]
    ids = {sp: {d.record(i).sample_id for i in range(len(d))} for sp, d in parts.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    assert sum(map(len, ids.values())) == 50
    assert not any(s.startswith("b/copy") or s == "b/resized.jpg" for v in ids.values() for s in v)


def test_finalize_reapplies_a_grown_blocklist_and_allows_missing_sources(tmp_path):
    """Benchmarks built after the sources (a grown blocklist) need only finalize again: exact SHA-1 and
    near-duplicate dHash matches are removed there. Missing sources make a corpus marked preliminary."""
    from onboard_vlm.data.hashing import sha1_bytes

    corpus, spec = _two_source_corpus(tmp_path)
    spec.write_text(spec.read_text() + "  - {dir: c, group: g, license: z}\n")  # never built
    with pytest.raises(AssertionError, match="allow_missing"):  # a missing source is an error unless allowed
        _run("finalize_corpus.py", "--corpus_root", str(corpus), "--sources_yaml", str(spec))
    raw_a = tmp_path / "raw" / "a"
    exact = sha1_bytes((raw_a / "07.jpg").read_bytes())  # stored unchanged (no downscale): same bytes
    near = dhash(_structured(20).resize((330, 250))) ^ 0b101  # a re-encoded, resized frame of a/10.jpg
    bl = tmp_path / "bl.json"
    json.dump({"sha1": [exact], "dhash": [f"{near:016x}"], "benchmarks": ["new_benchmark"], "eval_sets": []}, open(bl, "w"))
    _run("finalize_corpus.py", "--corpus_root", str(corpus), "--sources_yaml", str(spec), "--blocklist", str(bl), "--allow_missing")
    m = json.load(open(corpus / "corpus_manifest.json"))
    assert m["preliminary"] and m["missing_sources"] == ["c"]
    assert m["decontamination"]["removed"] == 2 and m["decontamination"]["benchmarks"] == ["new_benchmark"]
    assert m["sources"][0]["benchmark_dup_finalize"] == 2 and m["total_kept"] == 48
    kept = ImageCorpusDataset(str(corpus / "a"), split="all", require_finalized=True)
    ids = {kept.record(i).sample_id for i in range(len(kept))}
    assert "a/07.jpg" not in ids and "a/10.jpg" not in ids and len(ids) == 28
    assert "PRELIMINARY" in (corpus / "CORPUS_CARD.md").read_text()


def test_loader_splits_are_content_hashed(tmp_path):
    """Splits depend only on (image, fractions, seed): not on which shards / sources are loaded, and
    duplicates (same dHash) always land in the same split."""
    from onboard_vlm.data.corpus import assign_split

    corpus, _ = _two_source_corpus(tmp_path)  # not finalized: duplicates are still present
    kw = {"val_fraction": 0.3, "test_fraction": 0.2, "split_seed": 0}
    full = {sp: ImageCorpusDataset([str(corpus / "a"), str(corpus / "b")], split=sp, **kw) for sp in ("train", "val", "test")}
    lite = {sp: ImageCorpusDataset(str(corpus / "a"), split=sp, max_shards=2, **kw) for sp in ("train", "val", "test")}
    sid = lambda d: {d.record(i).sample_id for i in range(len(d))}  # noqa: E731
    for sp in ("train", "val", "test"):
        assert sid(lite[sp]) <= sid(full[sp])  # lite is a consistent subset
    where = {s: sp for sp, d in full.items() for s in sid(d)}
    for i in range(3):
        assert where[f"a/{i:02d}.jpg"] == where[f"b/copy{i}.jpg"]  # exact duplicate
    assert where["a/05.jpg"] == where["b/resized.jpg"]  # perceptual duplicate (same dHash)
    # statistics, determinism and the seed
    rng = np.random.default_rng(0)
    dh = rng.integers(1 << 20, 1 << 62, 100_000, dtype=np.uint64) | np.uint64(0xFFFF)
    sha = np.array([rng.bytes(20) for _ in range(100_000)], dtype="S20")
    c = assign_split(dh, sha, 0.05, 0.1, seed=0)
    assert abs((c == 1).mean() - 0.05) < 0.005 and abs((c == 2).mean() - 0.1) < 0.006
    assert (assign_split(dh, sha, 0.05, 0.1, seed=0) == c).all() and (assign_split(dh, sha, 0.05, 0.1, seed=1) != c).any()
    assert ((assign_split(dh, sha, 0.1, 0.1, seed=0) == 1) >= (c == 1)).all()  # growing val only moves train images
    with pytest.raises(ValueError):
        assign_split(dh, sha, 0.6, 0.5)


def test_data_module_rejects_leaky_split_configs(tmp_path):
    from onboard_vlm.data.data_module import DistillDataModule

    corpus, _ = _two_source_corpus(tmp_path)
    tr = [{"name": "image_corpus", "root": str(corpus / "a")}]
    with pytest.raises(ValueError, match="conflicting split"):
        DistillDataModule(tr, val=[{"name": "image_corpus", "root": str(corpus / "a"), "split": "val", "val_fraction": 0.5}])
    with pytest.raises(ValueError, match="reads split 'train'"):
        DistillDataModule(tr, val=[{"name": "image_corpus", "root": str(corpus / "a")}])
    with pytest.raises(ValueError, match="reads split 'test'"):
        DistillDataModule(tr, val=[{"name": "image_corpus", "root": str(corpus / "a"), "split": "test"}])
    dm = DistillDataModule(tr, val=[{"name": "image_corpus", "root": str(corpus / "a"), "split": "val", "group": "a"}],
                           test=[{"name": "image_corpus", "root": str(corpus / "a"), "split": "test", "group": "a"}],
                           dataset_defaults={"image_corpus": {"val_fraction": 0.3, "test_fraction": 0.3}}, num_workers=0, pin_memory=False)
    dm.setup()
    assert dm.val_groups == [("a", None)] and dm.test_groups == [("a", None)]  # legacy (no `evaluation`): one loader per group
    assert len(dm.train_set) + len(dm.val_sets["a"]) + len(dm.test_sets["a"]) == 30


def test_mixture_sampler_and_data_module_weights():
    from onboard_vlm.data.data_module import DistillDataModule
    from onboard_vlm.data.samplers import MixtureSampler

    s = MixtureSampler([10, 1000], [0.8, 0.2], 500, seed=0)
    idx = list(s)
    assert len(idx) == 500 and idx == list(s)
    assert 340 < sum(i < 10 for i in idx) < 460  # 0.8 of the draws from the 10-image component
    s.set_epoch(1)
    assert list(s) != idx
    train = [{"name": "synthetic", "num_samples": 10, "weight": 0.8, "group": "small"},
             {"name": "synthetic", "num_samples": 1000, "seed": 2, "weight": 0.2, "group": "large"}]
    dm = DistillDataModule(train, samples_per_epoch=200, num_workers=0, pin_memory=False, batch_size=4)
    dm.setup()
    sampler = dm.train_dataloader().sampler
    assert isinstance(sampler, MixtureSampler) and sampler.probs == [0.8, 0.2]
    dm = DistillDataModule(train, samples_per_epoch=200, num_workers=0, pin_memory=False, drop_groups=["small"])
    dm.setup()
    assert all(i >= 10 for i in dm.train_dataloader().sampler)
    with pytest.raises(ValueError, match="drop_groups"):
        DistillDataModule(train, drop_groups=["nope"])


def test_frame_corpus_excludes_benchmark_videos_and_static_frames(tmp_path):
    vids = tmp_path / "videos"
    vids.mkdir()
    write_video(str(vids / "keep_uid.mp4"), num_frames=60, fps=10, size=(320, 240))
    write_video(str(vids / "egoschema_src_uid.mp4"), num_frames=60, fps=10, size=(320, 240))
    bl = tmp_path / "bl.json"
    json.dump({"sha1": [], "dhash": [], "video_ids": ["egoschema_src_uid"], "video_spans": [["egoschema_src_uid", 0, 6]]}, open(bl, "w"))
    out = tmp_path / "frames"
    log = _run("make_frame_corpus.py", "--input_path", str(vids), "--fps", "2", "--name", "ego", "--output_path", str(out),
               "--blocklist", str(bl), "--still_bits", "0", "--min_std", "0")
    assert "excluded 1 benchmark source videos" in log
    ds = ImageCorpusDataset(str(out))
    assert len(ds) > 0 and all(ds[i]["sample_id"].startswith("ego/keep_uid/") for i in range(len(ds)))


def test_eval_blocklist_builder(tmp_path):
    from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord, encode_image

    img = Image.effect_mandelbrot((96, 64), (-2, -1.5, 1, 1.5), 30).convert("RGB")
    enc, hw = encode_image(img)
    with LMDBWriter(str(tmp_path / "bench"), "val", "bench") as w:
        w.add(VQARecord(sample_id="q1", dataset="bench", split="val", image=enc, question="?"), hw)
    with LMDBWriter(str(tmp_path / "bench"), "train", "bench") as w:  # train images stay usable
        w.add(VQARecord(sample_id="t1", dataset="bench", split="train", image=encode_image(Image.effect_noise((96, 64), 50).convert("RGB"))[0]), hw)
    with LMDBWriter(str(tmp_path / "textvqa"), "val", "textvqa") as w:  # TextVQA ids are Open Images ids
        w.add(VQARecord(sample_id="abc::q1", dataset="textvqa", split="val", image=enc, meta={"image_id": "abc"}), hw)
    uid_map = tmp_path / "uid_to_ego4d.json"
    json.dump({"clip1": {"video_uid": "ego4d_v1", "starting_sec": 240, "ending_sec": 420}}, open(uid_map, "w"))
    out = tmp_path / "bl.json"
    _run("make_eval_blocklist.py", "--lmdb_dirs", str(tmp_path / "bench"), str(tmp_path / "textvqa"), "--egoschema_uid_map", str(uid_map), "--output", str(out))
    bl = json.load(open(out))
    assert bl["dhash"] == [f"{dhash(enc):016x}"] and len(bl["sha1"]) == 1
    assert bl["source_ids"] == {"openimages": ["abc"]} and bl["counts"] == {"bench/val": 1, "textvqa/val": 0}
    assert bl["video_ids"] == ["ego4d_v1"] and bl["video_spans"] == [["ego4d_v1", 240, 420]]
