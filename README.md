# CamLab: multi-frame RAW burst experiments for the Galaxy A36

Two parts:

1. **Android app (`app/`)**: probes what the phone lets third-party apps do, and captures bursts
   (RAW DNG, or JPEG fallback) with **frozen exposure / ISO / white balance** and in-camera noise
   reduction and edge enhancement switched off where the device allows it.
2. **Python pipeline (`pipeline/`)**: align → robust merge → demosaic → colour → tone map →
   noise-aware denoise → gated sharpen, with every look-defining knob exposed, plus a learned
   kernel-prediction merger (PyTorch) you can train on your own bursts, and a stock-vs-ours comparison tool.

The split is deliberate: you iterate on ML and looks on a laptop/Colab in seconds, and only port
the winning stages to the phone (TFLite / native) later.

## Install the app

Every push to the branch builds an APK in GitHub Actions and publishes it to the
**`latest-debug` pre-release**. On the phone open
`https://github.com/VarunKumar-05/Samsung-camera-switch/releases/tag/latest-debug`,
download `app-debug.apk`, and install (allow "install unknown apps" for your browser when asked).
The APK is signed with a fixed key so later builds update in place.

## First run (do this before anything else)

1. Open CamLab, grant camera permission, tap **Probe device**. The report is shown on screen and saved to
   `Download/CamLab/probe/probe_<model>.txt`. It tells us whether the A36 exposes **RAW_SENSOR** to apps,
   at which sizes, the achievable burst rate, whether `MANUAL_SENSOR`/`BURST_CAPTURE` exist, and which
   Camera2/CameraX **extensions** (HDR / NIGHT / BOKEH / AUTO / FACE_RETOUCH) are available.
2. **Send me that report.** Samsung's mid-range phones often hide RAW from third-party apps; if so, the
   app falls back to JPEG bursts and I'll adapt the plan (the pipeline already has a JPEG path).
3. Toggle **Mode: RAW / JPEG**, choose a frame count, tap **CAPTURE BURST**. Hold still; use a tripod or a
   wall for the first tests. Output: `Download/CamLab/<timestamp>_raw/frame_000.dng … + meta.json`.
4. Take the **same scene** with the stock camera (default mode, and Pro/Expert RAW if you have it).

## Process on a computer

```bash
cd pipeline
pip install -r requirements.txt            # + torch for the ML part
# copy Download/CamLab/<session> from the phone (USB or Files app) to ./data/
python run_pipeline.py data/<session> --preset natural          # -> out/<session>_natural.png + _single.png
python run_pipeline.py data/<session> --set sharpen.amount=0.3 denoise.luma=1.5 tone.compress=0.6
python sweep.py data/<session> sharpen.amount 0 0.4 0.8 1.2     # contact sheet of one knob
python compare.py stock.jpg out/<session>_natural.png           # metrics + side-by-side + 100% crops
```

Presets: `natural` (default, restrained), `neutral` (nearly untouched, to judge the merge), `punchy`.
Knobs live in `camlab/params.py`.

### The ML part

`camlab/ml/kpn.py` is a small kernel-prediction network: it predicts per-pixel weights over a 3×3
neighbourhood of each aligned frame. The output is always a convex combination of real pixels, so it can
denoise and tolerate small misalignment/motion but cannot invent texture.

```bash
python -m camlab.ml.train --out kpn.pt --steps 1500                      # synthetic bootstrap
python -m camlab.ml.train --out kpn.pt --bursts data/s1 data/s2 data/s3  # + distil YOUR bursts
python run_pipeline.py data/<session> --ml kpn.pt
```
"Distillation" means: target = classical merge of *all* frames, input = only 4 of them. The model learns to
get close to the many-frame result from fewer frames, using your sensor's real noise.

## What is verified and what is not

- Verified by automated tests on synthetic bursts with known ground truth: Bayer pack/unpack, demosaic
  accuracy, sub-pixel alignment, noise reduction by merging (> +3 dB), ghost rejection for moving objects,
  that the sharpen/denoise knobs actually act, the JPEG fallback, and that the KPN trains and beats using
  only the reference frame.
- **Not yet verified on real A36 data**: DNG loading (`io.py`), the colour matrices from the phone's
  metadata, lens-shading correction, and whether the camera honours the frozen-exposure / NR-off requests.
  Expect a first-round of fixes once a real burst exists.
- The Android app compiles in CI, but has not been run on a device by me.
- A third-party app cannot reach Samsung's proprietary ISP; the realistic goal is *competitive and more
  natural*, not strictly better in every scene. The comparison tool is there so we find out with evidence.
