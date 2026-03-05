# HunyuanWorld-Voyager Summary

This document summarizes `HunyuanWorld-Voyager` based on a full read of the project paper (`HYWorld_Voyager.txt`) and codebase (`voyager/`, `data_engine/`, and top-level scripts), plus follow-up Q&A in this session.

## 1) What the project is

HunyuanWorld-Voyager is a geometry-conditioned video diffusion system for **single-image world exploration**.

Given:
- one input image,
- a camera trajectory,
- and a text prompt,

it generates a world-consistent RGB-D video clip (default 49 frames) that can be used for direct 3D reconstruction (for example, point cloud export).

## 2) What "world model" means here

In this project, "world model" is implemented as a generative pipeline that maintains spatial consistency through geometry-aware conditioning, not as a symbolic simulator.

- **State**: accumulated geometric context represented as point/depth-derived conditions.
- **Control**: camera trajectory and view-conditioned masks/partials.
- **Rollout**: iterative clip generation under trajectory constraints.

## 3) Main components

### A. World-consistent video diffusion
- Joint RGB+depth generation in one model.
- Core implementation:
  - `voyager/modules/models.py`
  - `voyager/inference.py`
  - `voyager/diffusion/pipelines/pipeline_hunyuan_video.py`
  - `voyager/diffusion/schedulers/scheduling_flow_match_discrete.py`

### B. Long-range world exploration
- Paper introduces world cache + point culling + smooth sampling.
- Open-source code mainly reflects this via condition generation and clip-level sampling interfaces rather than a single dedicated "cache manager" module.

### C. Scalable data engine
- Automatic camera/depth preparation from videos/images.
- Core scripts:
  - `data_engine/vggt_infer.py`
  - `data_engine/moge_infer.py`
  - `data_engine/metric3d_infer.py`
  - `data_engine/depth_align.py`
  - `data_engine/create_input.py`

## 4) Inference inputs and outputs

## Inputs expected by main sampler
`sample_image2video.py` consumes `--input-path` containing:
- `ref_image.png`
- `ref_depth.exr`
- `video_input/render_0000.png ... render_0048.png`
- `video_input/depth_0000.exr ... depth_0048.exr`
- `video_input/mask_0000.png ... mask_0048.png`

Plus CLI text prompt:
- `--prompt "..."`.

## Output
- Timestamped `.mp4` saved to `--save-path`.
- Video is RGB-D formatted by vertical stacking:
  - top half: RGB,
  - bottom half: depth visualization.
- Optional downstream:
  - convert generated RGB-D frame(s) to point cloud with `data_engine/convert_point.py`.

## 5) Training data shape (conceptual)

Training data consists of scene video clips and auto-annotated geometry:
- RGB frames,
- per-frame camera intrinsics/extrinsics,
- aligned depth maps.

Data sources described in paper:
- RealEstate10K,
- curated DL3DV subset,
- Unreal Engine renders,
- total scale: ~100k+ clips.

The model then uses rendered partial RGB/depth/masks as conditioning signals.

## 6) How the text prompt is used

## In training
- Text is tokenized/encoded through `TextEncoder` wrappers.
- Prompt templates are used (for image/video I2V prompt formats).
- Prompt dropout style conditioning is supported via training args (for CFG robustness).
- Embeddings are injected into DiT as `text_states` (and optional secondary text branch).

## In inference
- Prompt and negative prompt are encoded in pipeline.
- Classifier-free guidance combines unconditional and conditional predictions.
- Prompt affects semantic content/style while geometry conditions enforce spatial consistency.

## 7) Control signal clarification (important)

Based on paper + code:
- The current pipeline does **not** take explicit `W/S/A/D` action tokens as model inputs.
- It uses:
  - camera trajectory/path control,
  - partial RGB/depth/mask geometric conditions,
  - text prompt.

So movement is trajectory-based, not keyboard-token-based.

## 8) Useful entry points to read/run

Recommended read order:
1. `README.md`
2. `sample_image2video.py`
3. `voyager/inference.py`
4. `voyager/modules/models.py`
5. `voyager/diffusion/pipelines/pipeline_hunyuan_video.py`
6. `data_engine/create_input.py`

Operationally:
- Build condition assets first (`data_engine/create_input.py` or Gradio flow in `app.py`).
- Run generation (`sample_image2video.py`).

## 9) Practical constraints

- Hardware demand is high (README indicates ~60GB VRAM for 540p inference, with 80GB recommended).
- Distributed inference path is integrated via xDiT arguments (`--ulysses-degree`, `--ring-degree`).

---

If needed, this summary can be extended with:
- a symbol-level crosswalk from paper terms to exact classes/functions, or
- a tensor-shape trace for one full inference pass.
