# Design

## 1. Unified vision output contract

Every encoder, teacher or student, returns packed and unpadded tokens:

* `tokens [N, D]`, `intermediate_tokens {layer: [N, D_layer]}`, `image_token_lengths`.
* Images are concatenated in batch order.
* Within an image, tokens are row-major over its own `(h/48, w/48)` grid.

This is exactly the order of `Gemma4VisionModel`, whose pooler indexes kernels as
`x//3 + (W_patches//3) * (y//3)` and strips padding with a boolean mask. Students produce it with
`grid[valid_token_mask]` on a `[B, H/48, W/48, D]` map whose valid region is top-left aligned.

## 2. Geometry happens once

`geometry.sample_transform` returns a serialisable `TransformParams` (crop, flip, target size,
budget). `apply_transform` runs the crop, then the flip, then the Gemma resize: uint8, bicubic
and antialias, bit-identical to `Gemma4ImageProcessor`. The single uint8 result is then branched
into two views:

* **Teacher view**: patchified exactly as SigLIP2/Gemma and padded to `max_soft_tokens * 9`.
* **Student view**: a dense `[B, 3, H, W]` batch, zero-padded bottom/right, with `valid_token_mask`.

Stage 2 and evaluation use the official processor for text (chat template, image-token
expansion, `mm_token_type_ids`). They feed it the already-transformed images with
`do_resize=False`, so geometry is still done only once.

## 3. Students and 48-px alignment

| Student | Tokenisation | Batch invariant |
|---|---|---|
| `CompactViT` | 16-px patches, 2D-RoPE (Gemma's axial scheme), exact 3x3 average pooling, projection to `D_target` | yes (key-padding mask) |
| `CoarsePatchViT` | 48-px patches from a linear stem, or a mask-aware overlapping conv stem (strides 2-2-2-2-3) | yes |
| `MultiplexedViT` | as `CompactViT`, with a MiniViT trunk (section 4) | yes |
| `TimmSpatialStudent` | any `timm` `features_only` backbone with total stride 48 (below) | no; run per image at inference |

`TimmSpatialStudent` reaches stride 48 in one of two ways, both verified at construction on
non-square probes:

* **`stem_stride3`**: probes for every conv/pool that maps full resolution to 1/2 and sets those
  to stride 3. All parallel branches are covered, for example FastViT's MobileOne
  `conv_kxk` + `conv_scale`.
* **`tap16_conv3`**: takes the stride-16 stage and adds a learned 3x3/s3 conv.

`tests/test_timm_alignment.py` proves there is no drift. With circular padding, shifting the
image by 48 px shifts the token map by exactly one token (max error 0). The same test fails for
a stride-32 backbone followed by bilinear interpolation.

Intermediate stage maps at stride `s` are average-pooled by the exact factor `48/s`. There is
no interpolation anywhere.

## 4. Weight multiplexing (MiniViT)

`MultiplexedTrunk.shared` stores `num_shared_blocks` attention+MLP blocks once.
`MultiplexedTrunk.steps` holds one cheap `StepTransform` per invocation. Each step has its own
LayerNorms plus any of the following:

* `head_mix`: an `[H, H]` mixing of the post-softmax attention maps, computed as `A_h @ V_all`
  in one SDPA call, without materialising the maps.
* `dwconv`: a depthwise 3x3 conv over the token grid in the MLP hidden layer.
* `lowrank`: a rank-`r` adapter.
* `diag`: per-channel branch scales.

All transforms are initialised to identity, so step 0 of training equals plain weight sharing
(tested). The tests also check `numel`: shared parameters are counted once,
`named_parameters(remove_duplicate=False)` has no aliases, and the same module object runs
`steps_per_block` times.

`head_mix` multiplies the attention-value FLOPs by `num_heads`; `run_profile.py` shows the
cost. The `multiplexed_vit_s_lowrank` config is the FLOP-neutral variant.

## 5. Losses

All losses are composable as a weighted list in YAML (`config/criterion/*.yaml`).

* **`token_feature`**: MSE, Smooth-L1 or cosine on final tokens, after RMS normalisation by
  default. Gemma's `embed_vision` applies a scale-free RMSNorm before projecting into the LM,
  so per-token magnitude is invisible to the VLM.
* **`relational_token`**: for each image, `sim(X_i, X_i)` is matched with MSE or row-wise KL.
  There is never a global Gram matrix. Tests check that `loss([A, B]) == mean(loss([A]), loss([B]))`,
  that it is invariant to permutation, and that cross-image gradients are zero.
* **`intermediate_layer`**: feature or relational matching on the layers in `layer_map`.
* **`grafted_logit_kd`**: temperature-scaled `KL(teacher || grafted)` on the answer span only.
  `logits_to_keep` keeps the 262K-vocabulary tensor to `[B, K, V]` with `K = max answer length + 1`.

## 6. Cross-dimension adapter

`DistillationAdapter.from_dims(student, teacher_dim, layer_map)` reads the teacher width from
`config.vision_config.hidden_size`. It adds `Linear` heads (equivalent to 1x1 convs on the token
grid) only where widths differ, or everywhere with `force_*`. It belongs to the task, not the
student, so `export_student()` never contains it.

## 7. Grafting and the two stages

`StudentVisionTower` implements the `Gemma4VisionModel` call signature
(`pixel_values, pixel_position_ids -> last_hidden_state [N, D]`). It reconstructs dense images
losslessly from the processor's patches, so `get_image_features` and `get_video_features`, and
therefore `generate()`, work unchanged.

* **Stage 1** (`Stage1FeatureDistillTask`): the frozen teacher lives outside the module tree, so
  it is never checkpointed, optimised or DDP-wrapped. It can be replaced by the safetensors
  feature cache.
* **Stage 2** (`Stage2GraftedKDTask`): the student is grafted into the deployment VLM; only
  `embed_vision` (full or LoRA) and optionally the student are trainable. Target logits come
  from the same VLM through `with grafted.teacher_vision():`, or from a larger Gemma 4 that
  shares the tokenizer. Checkpoints hold only trainable tensors. The export contains
  `student.safetensors` and `embed_vision.safetensors` with LoRA merged.
