# Particle Diameter Analyzer

Measure the diameter of particles / droplets in microscopy images and export an
annotated image, a size-distribution histogram, and a combined CSV of every
measurement. Scale is calibrated from a single reference ("original") image or
from a known micron-per-pixel ratio, and reused across a whole batch.

Built for magenta / pink droplets on a grey background (e.g. optical-microscope
images of microfluidic droplets), but the parameters are exposed at the top of
the script so it can be adapted to other particle types.

## Features

- **Two detection methods** — a Hough circle transform (default; best for large,
  dense, *overlapping* droplets) and a watershed segmentation (best for small,
  well-separated droplets).
- **Flexible scale calibration** — direct µm/px ratio, a manually measured
  scale-bar length, one reference image whose scale bar is detected once and
  applied to all images, or automatic per-image detection.
- **Equivalent circular diameter (ECD)** for the watershed method, and rim-fitted
  radii for Hough, with phantom-ring and abnormal-radius rejection.
- **Batch processing** — one failing image never stops the run.
- **Outputs** — per-image annotated images, a combined (or per-image) histogram
  with a Gaussian fit, and a single CSV with mean / SD / CV statistics.

## Installation

Requires Python 3.8+.

```bash
git clone https://github.com/<your-username>/particle-diameter-analyzer.git
cd particle-diameter-analyzer
pip install -r requirements.txt
```

## Usage

1. Put your images in the `images/` folder (or set `INPUT_DIR` / `IMAGE_PATHS`
   at the top of `diameter_calculation.py`).
2. Set the scale (see below) and, if needed, adjust the detection parameters.
3. Run:

```bash
python diameter_calculation.py
```

Results are written to the `output/` folder:

| File | Description |
|------|-------------|
| `circles_<image>.png` | original image with fitted circles + index numbers |
| `histogram_combined.png` | pooled size distribution with Gaussian fit and statistics |
| `particle_diameters_all.csv` | every measurement (diameter in µm and px, circularity, centre) |

## Scale calibration

The script resolves the scale in this priority order (first one that is set wins):

1. **`SCALE_UM_PER_PX`** — the direct µm-per-pixel ratio. Use this when you know
   it from microscope calibration, or when the image has no scale bar.
2. **`SCALE_BAR_PX`** — the scale-bar length in pixels, measured once by hand.
3. **`SCALE_BAR_IMAGE`** — a reference image; its scale bar is detected once and
   applied to every image. Only valid when all images share the same magnification.
4. **`SHARED_SCALE_BAR = True`** — auto-detect the scale bar from the first image
   where one is found and reuse it for the batch (default).

> **Important:** each magnification has its own µm/px value. Do not reuse a value
> across images taken at different magnifications or resolutions.

`SCALE_BAR_UM` sets the real length the bar represents (default `100.0` µm).

## Key parameters

All parameters live in the `CONFIGURATION` block at the top of the script. The
most commonly adjusted ones:

| Parameter | Purpose |
|-----------|---------|
| `DETECTION_METHOD` | `"hough"` or `"watershed"` |
| `MIN_PARTICLE_AREA_PX` | drop blobs smaller than this (debris / noise) |
| `HOUGH_RADIUS_PX` | droplet radius range in px; `(None, None)` = auto |
| `HOUGH_PARAM2` | Hough roundness threshold (lower = more circles) |
| `MIN_DISTANCE_PEAKS` | minimum spacing between centres when splitting (watershed) |
| `N_BINS` | number of histogram bars |
| `COMBINED_HISTOGRAM` | pool all images into one histogram |

Each parameter is documented inline in `diameter_calculation.py`.

## License

Released under the MIT License — see [LICENSE](LICENSE).
