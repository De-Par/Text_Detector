# Text Detector 🚀 (DBNet / PP-OCR det on ONNX Runtime, CPU-only)

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![OpenCV 4](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)](https://opencv.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU--only-1f6feb.svg)](https://onnxruntime.ai/)
[![OpenMP](https://img.shields.io/badge/OpenMP-enabled-2ea44f.svg)](https://www.openmp.org/)
[![Meson + Ninja](https://img.shields.io/badge/build-Meson%20%2B%20Ninja-ff69b4.svg)](https://mesonbuild.com/)
[![macOS / Linux](https://img.shields.io/badge/OS-macOS%20%7C%20Linux-lightgrey.svg)](#)

![plot](img/demo.png)

A fast, **CPU-only** text detector powered by ONNX Runtime. It supports **tiled inference**, **polygon NMS**, **IOBinding** (to eliminate per-frame allocations), and a **benchmark mode** with p50/p90/p99 latency reporting. Designed for production: clean code, robust shape handling (NCHW/NHWC/2D/3D) and safe defaults for multi-core servers. Output contains image with quadrilateral boxes + 4 points *(x, y)* of each box printed to **stdout**.


## Table of Contents

- [Highlights](#highlights)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Install & Build](#install--build)
- [Model Zoo](#model-zoo)
- [Command-line Options](#command-line-options)
- [Quick Start](#quick-start)
- [Common Recipes](#common-recipes)
- [Performance Tuning Guide](#performance-tuning-guide)
- [Benchmark Mode](#benchmark-mode)
- [IOBinding Deep-Dive](#iobinding-deep-dive)
- [Tiling & NMS](#tiling--nms)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [License](#license)


## Highlights

- ⚡ **Fast CPU inference** (x86 / ARM, macOS & Linux)
- 🧩 **Tiled inference** (RxC grid) with overlap + **polygonal NMS**
- 💾 **IOBinding**: reuse input/output buffers, **zero allocations per frame**
- 📈 **Bench mode**: p50/p90/p99 latency, warmup, optional no-draw
- 🧠 Robust output shape support: `[1,1,H,W]`, `[1,H,W,1]`, `[1,H,W]`, `[H,W]`
- 🔒 **Threading done right**: separate knobs for **OpenMP** (tiles) and **ORT** (intra-op)
- 🧪 Clean logging: detections to **stdout**, performance to **stderr**


## How it works

```
File → OpenCV decode (BGR8)
     → Resize (dynamic --side or fixed --fixed_hw)
     → Normalize (RGB float32, CHW)
     → ONNX Runtime (backbone/neck/head) → probability map (or logits)
     → (optional) Sigmoid (--apply_sigmoid 1)
     → Threshold + morphology (--unclip)
     → Contours → minAreaRect → ordered quad
     → Map coords back to original image size
     → (Tiles) offset + polygon NMS
     → Draw boxes, print coordinates to stdout
```

> 💡 **Why separate thread knobs?**  
> - `--tile_omp` (OpenMP) parallelizes **across tiles** (outer level).  
> - `--threads` (ONNX Runtime) parallelizes **within a single tile** (intra-op).  
> On big CPUs, use **many OMP threads** and **few ORT threads** (often 1–2) to avoid oversubscription.


## Requirements

- **C++17**, **Meson** (≥ 1.0), **Ninja**, **pkg-config**
- **OpenCV 4.x** (core, imgproc, imgcodecs)
- **ONNX Runtime** (CPU EP)
- **OpenMP** (recommended for tiling)

### Linux (Ubuntu example)

```bash
sudo apt-get update
sudo apt-get install -y build-essential ninja-build meson cmake cmake-data pkg-config libopencv-dev python3 python3-pip libomp-dev 
```

#### Install ONNX Runtime:

Either use official binaries (copy headers+libs into /usr/local) or build from source (Release, CPU only):

```bash
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel
```

After build finishes, copy headers+libs to /usr/local (adjust paths if needed):
```bash
sudo cp -r include/onnxruntime /usr/local/include/
sudo cp -d build/Linux/Release/libonnxruntime.so* /usr/local/lib/
sudo cp -d build/Linux/Release/libonnxruntime_providers_shared.so /usr/local/lib/
sudo ldconfig
```

---

### MacOS (Apple Silicon)

```bash
brew install meson ninja opencv onnxruntime libomp
```
Headers typically at `/opt/homebrew/Cellar/onnxruntime/<version>/include` and libraries at `/opt/homebrew/Cellar/onnxruntime/<version>/lib`


## Install & Build

```bash
meson setup builddir
meson compile -C builddir 
```

> 💡 If you see `onnxruntime_cxx_api.h: No such file or directory`, verify that ORT headers are discoverable by Meson (e.g., Homebrew path `/opt/homebrew/Cellar/onnxruntime/<version>/include` on MacOS).


## Model Zoo 

This project is **model-agnostic** as long as your detector exports a single-channel probability (or logit) map. Below are two practical sources of ready-to-use models.

### 1) MMOCR (PyTorch) models → ONNX

MMOCR provides many DBNet-based detectors (R50, MobileNet, DCN variants, etc.). You can export them to ONNX and use them directly with this tool. Detailed information about available models you can find there: [mmocr_models](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html?utm_source=chatgpt.com). Also, take a look on support in ONNX Runtime: [mmocr_support](https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmocr.html).

**Export with MMOCR’s `pytorch2onnx.py`**

1) Clone and install MMOCR (use versions compatible with your checkpoint):
```bash
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
python3.11 -m venv mvenv
source ./mvenv/bin/activate
pip install -r requirements.txt
pip install onnx onnxsim 
```

2) Export to ONNX:
```bash
python tools/deployment/pytorch2onnx.py <CONFIG.py> --checkpoint <MODEL.pth> --output-file <OUT.onnx> --opset 11 --dynamic-export
```

3) (Optional) Simplify the graph:
```bash
python -m onnxsim <OUT.onnx> <OUT-sim.onnx>
```

**Notes & tips**

- Prefer **opset ≥ 11**. For CPU inference, 11–13 is typically safe.
- If you need dynamic spatial sizes, keep `--dynamic-export`; otherwise static shapes plus `--fixed_hw` may be faster/stabler.
- Some MMOCR configs already include the final **Sigmoid** in the head. If your output looks like logits, run with `--apply_sigmoid 1`.
- Keep input channels at 3 unless you **change the first conv to 1-channel** and re-train/fine-tune (grayscale alone rarely gives a big speedup).

If you prefer **MMDeploy**, you can export via MMDeploy’s ONNX pipeline as well: just ensure the resulting model outputs a single-channel map and that pre/post-processing matches what this app expects.

### 2) PaddleOCR ONNX

There are pre-converted **PaddleOCR** detectors on the Hugging Face Hub: [deepghs/paddleocr](https://huggingface.co/deepghs/paddleocr/tree/main), including lightweight **PP-OCR mobile** variants. Typical file names include:

- `ch_PP-OCRv3_det_infer.onnx`
- `en_PP-OCRv3_det_infer.onnx`
- `ch_ppocr_mobile_v2.0_det_infer.onnx`
- `ch_ppocr_mobile_slim_v2_det.onnx`
- `...`

**Important compatibility notes**

- **Output often contains logits** → run with `--apply_sigmoid 1`.
- **Normalization differs from ImageNet**: PaddleOCR commonly uses `img = (img/255.0 - 0.5) / 0.5` (i.e., `mean=(0.5,0.5,0.5)`, `std=(0.5,0.5,0.5)`).  
  The current code uses ImageNet stats (`mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`). For best accuracy with Paddle models, **adjust the normalization in code** to Paddle’s scheme or re-export to match ImageNet stats.
- **Input sizes** are typically dynamic with the constraint **H,W % 32 == 0**. Use `--fixed_hw` (e.g., `640x640`) or `--side` to meet that requirement.
- If you see `Unexpected output shape`, your detector might output a different tensor layout. This app handles `[1,1,H,W]`, `[1,H,W,1]`, `[1,H,W]`, and `[H,W]`. If yours differs, inspect the model head or adjust the post-processing accordingly.

**Quick start with a PaddleOCR ONNX**

```bash
./text_det --model det.onnx --image input.png --apply_sigmoid 1 --fixed_hw 640x640 --threads 1
```

> 💡 If you switch to Paddle normalization, update mean / std in code accordingly.

> 💡 For highest stability in batch/production (hundreds of images): combine **IOBinding** (`--bind_io 1`) with a **fixed input size** (`--fixed_hw WxH`) and keep ORT threads small (`--threads 1–2`) while scaling tiles via OpenMP (`--tile_omp`).

---

## Command-line Options

| Flag | Type | Default | Description |
|---|---|---:|---|
| `--model` | string | — | Path to ONNX detector (DBNet / PP-OCR det). |
| `--image` | string | — | Path to input image. |
| `--out` | string | `out.png` | Output image with drawn boxes. |
| `--bin_thresh` | float | `0.3` | Threshold for binarizing **probability map** (0..1). |
| `--box_thresh` | float | `0.6` | Filter boxes by mean probability inside polygon. |
| `--side` | int | `960` | Max side length (dynamic resize, keep aspect; rounded to multiple of 32). Ignored if `--fixed_hw` is set. |
| `--threads` | int | `0→1` | **ONNX Runtime intra-op** threads **per tile**. Use 1–2 with tiling. |
| `--unclip` | float | `1.5` | Morphological “inflate” before contours (DB-style). |
| `--apply_sigmoid` | 0/1 | `0` | Apply sigmoid if model outputs logits (not in [0,1]). |
| `--tiles` | `RxC` | — | Enable tiling (e.g., `3x3`). Each tile runs inference separately. |
| `--tile_overlap` | float | `0.10` | Fractional overlap for tiles (0..0.5) to avoid cut words. |
| `--nms_iou` | float | `0.30` | Polygon NMS IoU threshold to drop duplicates between tiles. |
| `--tile_omp` | int | `0→env/auto` | **OpenMP** threads for **tile-level** parallelism. |
| `--omp_places` | string | `cores` | Sets `OMP_PLACES` (e.g., `cores`, `threads`, `sockets`, or custom `{…}`). |
| `--omp_bind` | string | `close` | Sets `OMP_PROC_BIND` (`close`, `spread`, `master`, `true`, `false`). |
| `--bind_io` | 0/1 | `0` | Enable **IOBinding** (reuses buffers; no per-frame allocations). |
| `--fixed_hw` | `WxH` | — | Fixed input size (e.g., `640x640`, rounded to /32). Great with `--bind_io`. |
| `--bench` | int | — | Run benchmark for N iterations (p50/p90/p99). |
| `--warmup` | int | `20` | Warmup iterations (excluded from stats). |
| `--no_draw` | 0/1 | `0` | In bench mode, disable drawing/saving to keep timings clean. |
| `-h`, `--help` | — | — | Show usage. |

⚠️ Output format (stdout) is one line per detection (vertices are in consistent clockwise order):
```
x0,y0 x1,y1 x2,y2 x3,y3
```


## Quick Start

**Basic (no tiling):**
```bash
./text_det --model det.onnx --image input.png --threads 4 --side 640 --bin_thresh 0.3 --box_thresh 0.6
```

**Model that outputs logits (no final Sigmoid):**
```bash
./text_det --model det.onnx --image input.png --threads 4 --apply_sigmoid 1 --bin_thresh 0.3 --box_thresh 0.3
```


## Common Recipes

**Tiling on a big server (e.g., 96 cores)**
```bash
./text_det --model det.onnx --image input.png --tiles 3x3 --tile_overlap 0.15 --nms_iou 0.3 --threads 2 --tile_omp 8 --omp_places cores --omp_bind close
```
- Keep **ORT intra-op small** (`--threads 1–2`).
- Use **lots of OpenMP threads for tiles** (`--tile_omp`).

**IOBinding + fixed size (best reuse, hundreds of images)**
```bash
./text_det --model det.onnx --image input.png --bind_io 1 --fixed_hw 640x640 --threads 4
```

**Tiling + IOBinding + fixed size (stable latency under load)**
```bash
./text_det --model det.onnx --image input.png --tiles 3x3 --tile_overlap 0.15 --nms_iou 0.3 --bind_io 1 --fixed_hw 640x640 --threads 2 --tile_omp 8 --omp_places cores --omp_bind close
```


## Performance Tuning Guide

- **Two levels of parallelism**:
  - **OpenMP (outer)** = `--tile_omp` (or `OMP_NUM_THREADS`) → parallel tiles.
  - **ONNX Runtime (inner)** = `--threads` → parallel inside a tile.
- **Avoid oversubscription**: on large CPUs, prefer **many tiles** (`--tile_omp`) and **few ORT threads** (`--threads 1–2`).
- **Pin threads for cache locality**:
  - `--omp_places cores` + `--omp_bind close` is a safe default.
  - Dual-socket NUMA? Try `--omp_bind spread`.
- **IOBinding**:
  - Enable `--bind_io 1`; ideally combine with `--fixed_hw WxH` (multiple of 32) to **never re-bind**.
- **Thresholds**:
  - `--bin_thresh` usually 0.2–0.4, `--box_thresh` 0.5–0.7.
  - For small text, increase `--side` or use tiling with overlap `0.10–0.20`.


## Benchmark Mode

Measure end-to-end latency with warmup and tail-latency percentiles:
```bash
./text_det --model det.onnx --image input.png --tiles 3x3 --tile_overlap 0.15 --nms_iou 0.3 --bind_io 1 --fixed_hw 640x640 --threads 2 --tile_omp 8 --bench 200 --warmup 50 --no_draw 1
```

**Report includes (stderr):**
- `total_ms`: avg, **p50**, **p90**, **p99** (entire pipeline),
- `infer_ms`: **p50**, **p90**, **p99** (sum of ORT time across tiles),
- `fps@p50`: quick throughput estimate at median.

> 💡 Tip: For consistent numbers, disable drawing/saving (`--no_draw 1`) and keep shapes fixed (`--fixed_hw`).


## IOBinding Deep-Dive

**What it is**: binding ONNX input / output tensors directly to your **pre-allocated** buffers.  
**Why it matters**: eliminates per-frame allocations & copies, improving latency stability.

**Best practice**:
- Set `--bind_io 1`.
- Use **fixed shapes** with `--fixed_hw WxH` (rounded to /32).
- With tiling, each OpenMP worker gets its **own binding context** (no locks).

> 💡 Without `--fixed_hw`, the code will **probe once per new size** (first call), bind, and then reuse for that WxH in that worker.


## Tiling & NMS

- `--tiles RxC` splits the image into a grid and runs inference per tile.  
- `--tile_overlap` avoids cutting words at tile borders.  
- After stitching, **polygon NMS** removes duplicate boxes across tiles using IoU (typical `0.2–0.4`).

> 💡 For heavy servers: tiling scales extremely well with OpenMP (outer) threads. Keep ORT threads small.


## Troubleshooting

- **`onnxruntime_cxx_api.h: No such file or directory`**  
Make sure ONNX Runtime is installed and headers are visible to Meson (e.g., `/usr/local/include` on Linux, `/opt/homebrew/opt/onnxruntime/include` on macOS).

- **`Unexpected output shape`**  
This tool supports `[1,1,H,W]`, `[1,H,W,1]`, `[1,H,W]`, `[H,W]`. If your model differs, verify your export and the final layers. If outputs are **logits** (not in [0,1]), pass `--apply_sigmoid 1`.

- **Performance flatlines when increasing threads**  
Likely oversubscription. Lower `--threads` (ORT) to 1–2; increase `--tile_omp`; pin threads: `--omp_places cores --omp_bind close`.

- **Boxes are weak or too many false positives**  
Tune `--bin_thresh`, `--box_thresh`, `--unclip`. If model lacks final sigmoid, set `--apply_sigmoid 1`.


## FAQ

**Q: Can I speed up by feeding grayscale instead of RGB?**  
Not unless the **model itself** is changed to accept `[1,1,H,W]`. Feeding one channel into `[1,3,H,W]` doesn’t reduce compute. Changing the first conv to 1-channel helps only a little overall; accuracy may drop.

**Q: How are coordinates printed?**  
Each detection line on **stdout**: `x0,y0 x1,y1 x2,y2 x3,y3` (ordered clockwise).

**Q: Does the tool support dynamic sizes?**  
Yes. Dynamic path uses `--side`. For best latency and zero re-binding, prefer `--fixed_hw WxH` with `--bind_io 1`.


## Roadmap

- Optional **AABB/connected-components** fast postprocess mode
- Optional **micro-batch tiling** (pack multiple tiles into a single `N×C×H×W` run)
- Built-in accuracy eval (precision/recall/F1) against custom annotation formats
- ...


## License

MIT - feel free to change for your repo’s needs.


### Credits

This project uses OpenCV, OpenMP and ONNX Runtime. Model families supported include DBNet and PP-OCR det models exported to ONNX.


👾 **Happy detecting!** 👾
