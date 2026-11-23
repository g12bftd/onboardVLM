"""Verify that the teacher is *exactly* the Gemma 4 E2B / E4B vision encoder.

    # (re)build the bundled manifest from the four pinned checkpoints (range reads, ~320 MB each)
    python scripts/verify_gemma4_teacher.py --write_manifest

    # verify the exact-weights loader for one checkpoint (fast: vision tensors only)
    python scripts/verify_gemma4_teacher.py --model_id google/gemma-4-E2B-it

    # strongest check: also load the FULL VLM with HF from_pretrained (downloads 10-16 GB, needs
    # ~12 GB RAM for E2B) and require bitwise-identical weights AND outputs, through both the raw
    # tower and our teacher + collator path vs. the official processor + get_image_features.
    python scripts/verify_gemma4_teacher.py --model_id google/gemma-4-E2B-it --full_check
"""

from __future__ import annotations

import argparse
import gc
import json

import torch

from onboard_vlm.teachers import hub


def write_manifest() -> None:
    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi()
    vision_ref: dict[str, dict] | None = None
    checkpoints = {}
    for model_id, spec in hub.SUPPORTED_TEACHERS.items():
        with HfFileSystem().open(f"{model_id}/model.safetensors", "rb", revision=spec.revision, block_size=64 << 20) as f:
            tensors, hashes, header = hub.read_safetensors_subset(f, hub._keep)
        vis = {
            k[len(hub.VISION_PREFIX) :]: {"sha256": hashes[k], "dtype": header[k]["dtype"], "shape": header[k]["shape"]}
            for k in hashes if k.startswith(hub.VISION_PREFIX)
        }
        if vision_ref is None:
            vision_ref = vis
        elif vis != vision_ref:
            diff = [k for k in vis if vis[k] != vision_ref.get(k)]
            print(f"NOTE: {model_id} vision tower differs from the first checkpoint in {len(diff)} tensors")
        info = api.model_info(model_id, revision=spec.revision, files_metadata=True)
        sib = next(s for s in info.siblings if s.rfilename == "model.safetensors")
        e = header[hub.EMBED_VISION_KEY]
        checkpoints[model_id] = {
            "revision": spec.revision, "size": spec.size, "instruction_tuned": spec.instruction_tuned,
            "file": "model.safetensors", "file_size": sib.size, "file_sha256": sib.lfs.sha256 if sib.lfs else None,
            "text_hidden_size": spec.text_hidden_size,
            "vision_tower_digest": hub.set_digest({k: v["sha256"] for k, v in vis.items()}),
            "embed_vision": {"sha256": hashes[hub.EMBED_VISION_KEY], "dtype": e["dtype"], "shape": e["shape"]},
        }
        print(model_id, "vision digest", checkpoints[model_id]["vision_tower_digest"][:16], "embed_vision", hashes[hub.EMBED_VISION_KEY][:16])
        del tensors
    assert vision_ref is not None
    manifest = {
        "description": "SHA-256 of the raw checkpoint bytes of every Gemma 4 E2B/E4B vision-tower tensor and of embed_vision.",
        "vision_tower": {
            "num_tensors": len(vision_ref),
            "num_elements": int(sum(torch.Size(v["shape"]).numel() for v in vision_ref.values())),
            "digest": hub.set_digest({k: v["sha256"] for k, v in vision_ref.items()}),
            "identical_across_checkpoints": len({c["vision_tower_digest"] for c in checkpoints.values()}) == 1,
            "tensors": dict(sorted(vision_ref.items())),
        },
        "checkpoints": checkpoints,
    }
    hub.MANIFEST_PATH.parent.mkdir(exist_ok=True)
    hub.MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", hub.MANIFEST_PATH)


def _images():
    g = torch.Generator().manual_seed(0)
    return [torch.randint(0, 256, (3, h, w), dtype=torch.uint8, generator=g) for h, w in [(480, 640), (1080, 1920), (300, 1200), (97, 131)]]


@torch.no_grad()
def verify(model_id: str, full_check: bool, budgets: list[int]) -> dict:
    from onboard_vlm.data.collator import DualViewCollator
    from onboard_vlm.teachers.gemma4 import Gemma4Teacher

    report: dict = {"model_id": model_id}
    teacher = Gemma4Teacher(model_id, load="vision", verify=True)
    report["exact_weights_vs_manifest"] = True  # Gemma4Teacher raises otherwise
    if not full_check:
        return report

    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    spec = hub.resolve_teacher(model_id)
    vlm = Gemma4ForConditionalGeneration.from_pretrained(model_id, revision=spec.revision, dtype=torch.bfloat16)
    hf_sd, ours_sd = vlm.model.vision_tower.state_dict(), teacher.vision_tower.state_dict()
    report["weights_bitwise_equal_to_from_pretrained"] = set(hf_sd) == set(ours_sd) and all(torch.equal(hf_sd[k], ours_sd[k]) and hf_sd[k].dtype == ours_sd[k].dtype for k in hf_sd)
    report["embed_vision_bitwise_equal"] = torch.equal(vlm.model.embed_vision.embedding_projection.weight, teacher.embed_projection.weight)
    proc = AutoProcessor.from_pretrained(model_id, revision=spec.revision)
    imgs = _images()
    for budget in budgets:
        x = proc.image_processor(imgs, return_tensors="pt", max_soft_tokens=budget)
        ref = vlm.model.get_image_features(pixel_values=x["pixel_values"].to(torch.bfloat16), image_position_ids=x["image_position_ids"])
        batch = DualViewCollator(budget, train=False)([{"sample_id": str(i), "image": im} for i, im in enumerate(imgs)])
        ours = teacher.encode_vision(batch.teacher_inputs)
        ref_lm = torch.cat(ref.pooler_output)
        ours_lm = teacher.project_to_lm(ours.tokens)
        report[f"budget_{budget}"] = {
            "num_tokens": int(ours.tokens.shape[0]),
            "inputs_bitwise_equal": torch.equal(batch.teacher_inputs.pixel_values, x["pixel_values"]) and torch.equal(batch.teacher_inputs.pixel_position_ids, x["image_position_ids"]),
            "vision_tokens_bitwise_equal": torch.equal(ref.last_hidden_state, ours.tokens),
            "lm_embeddings_bitwise_equal": torch.equal(ref_lm, ours_lm),
        }
    del vlm
    gc.collect()
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write_manifest", action="store_true")
    p.add_argument("--model_id", default=None, choices=sorted(hub.SUPPORTED_TEACHERS))
    p.add_argument("--full_check", action="store_true")
    p.add_argument("--budgets", type=int, nargs="+", default=[70, 280, 1120])
    args = p.parse_args()
    if args.write_manifest:
        write_manifest()
    if args.model_id:
        report = verify(args.model_id, args.full_check, args.budgets)
        print(json.dumps(report, indent=2))
        flat = [v for v in report.values() if isinstance(v, bool)] + [x for v in report.values() if isinstance(v, dict) for x in v.values() if isinstance(x, bool)]
        if not all(flat):
            raise SystemExit("VERIFICATION FAILED")
        print("VERIFIED: teacher is bitwise identical to the official checkpoint")


if __name__ == "__main__":
    main()
