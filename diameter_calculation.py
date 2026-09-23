"""
Particle Diameter Analyzer
==========================
Measure the diameter of particles / droplets in microscopy images and export:
  * an annotated image with the fitted circles drawn on top,
  * a size-distribution histogram (with a Gaussian fit), and
  * a combined CSV table of every measurement.

Scale calibration can come from any of the following (highest priority first):
  1. a direct micron-per-pixel ratio,
  2. a manually measured scale-bar length in pixels,
  3. a reference ("original") image whose scale bar is detected once and
     applied to every image,
  4. automatic per-image (or shared) scale-bar detection.

Two detection methods are available: a Hough circle transform (best for large,
dense, overlapping droplets) and a watershed segmentation (best for small,
well-separated droplets on a clear background).
"""

import csv
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ============================================================
#  CONFIGURATION - EDIT THESE TO MATCH YOUR IMAGES
# ============================================================
# --- IMAGE SOURCE: choose one of the two options below ---
# Option A (default): scan EVERY image inside one folder.
INPUT_DIR = "images"
# Option B: list each image explicitly. When used, set IMAGE_PATHS = [...] and
# INPUT_DIR is ignored.
IMAGE_PATHS = None
# example:
# IMAGE_PATHS = [
#     "images/img1.png",
#     "images/img2.png",
# ]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

SEARCH_SUBFOLDERS = False           # True = also look for images inside sub-folders of INPUT_DIR
                                    # (enable this if your images live in several sub-folders).

# --- Scale calibration ---
# HIGHEST PRIORITY: enter the micron-per-pixel ratio directly if you know it (e.g. from
# microscope calibration, or when the image has NO scale bar). When a number is set here,
# EVERY scale-bar detection mechanism below is skipped.
# ***IMPORTANT***: each MAGNIFICATION has its own micron/px value. Do NOT reuse one value
# across images taken at different magnifications / resolutions (that is what makes a
# diameter come out as, say, ~800 um by mistake).
SCALE_UM_PER_PX = None              # e.g. 0.79 (means 1 pixel = 0.79 um). None = use a scale bar.

SCALE_BAR_UM = 100.0                # real length of the scale bar (image is labelled "100 um").
SCALE_BAR_PX = None                 # None = auto-detect. If set (measured once by hand), every image uses it.

# Reference ("original") image used to read the scale bar: detect the bar on THIS image and
# apply it to every other image. Only valid when this reference shares the SAME magnification
# as the images being measured.
# May be a full path or just a file name (the code will look for it inside INPUT_DIR).
# Leave as None to use the automatic mechanism (SHARED_SCALE_BAR) below.
SCALE_BAR_IMAGE = None

SHARED_SCALE_BAR = True             # (used only when SCALE_UM_PER_PX / SCALE_BAR_PX / SCALE_BAR_IMAGE are all None)
                                    #   True  = auto-detect the scale bar from the first image where it is found
                                    #           and reuse it for all images.
                                    #   False = detect per image (images where detection fails are skipped).

# --- Particle filters ---
MIN_PARTICLE_AREA_PX = 300          # drop blobs smaller than this (debris / noise), in pixel^2.
MAX_GREEN_CHANNEL = 120             # pink / magenta droplets have a low Green channel -> reject off-colour objects.
                                    # (droplets in the sample images have green ~8, so 120 is very safe)
EXCLUDE_EDGE_PARTICLES = True       # True = drop droplets cut off at the image border (diameter not measurable).
                                    # Set False to also circle border droplets (less accurate).

# --- DROPLET DETECTION METHOD ---
DETECTION_METHOD = "hough"          # "hough" (RECOMMENDED for large, dense, OVERLAPPING droplets - fits circles
                                    #   to the curved rim and can separate overlapping droplets), or
                                    # "watershed" (for small, well-separated droplets on a clearly open background).

# Droplet radius range (px) for Hough. (None, None) = AUTO-estimate per image (recommended).
# If auto is not accurate, set by hand: e.g. droplets ~340px in diameter -> radius ~170 -> use (120, 230).
HOUGH_RADIUS_PX = (None, None)
HOUGH_PARAM2 = 30                   # Hough "roundness" threshold: LOWER -> catch more circles (more false rings),
                                    # HIGHER -> stricter (more misses). 28-40 is a reasonable range.
HOUGH_MIN_EDGE_SUPPORT = 0.35       # reject bad fits: at least this fraction of the circumference must land on
                                    # the droplet's dark rim.
                                    # increase (0.5) if you see extra rings; decrease (0.25) if droplets are missed.
HOUGH_DARK_THRESH = None            # "dark rim" threshold (0-255). None = AUTO per image
                                    # (= 0.7 x magenta interior brightness, which sits between the dark rim and
                                    # the bright core). Only set a fixed number if the auto value is off.
HOUGH_MAX_INNER_DARK = 0.10         # REJECT PHANTOM RINGS that span several droplets: a real droplet has a clean
                                    # core (few dark pixels), whereas a phantom ring sits over the gap between
                                    # droplets and has a dark rim crossing its core.
                                    # If the dark-pixel fraction inside the core (45% of the radius) exceeds this
                                    # value -> reject.
                                    # increase (0.15) if real droplets are being rejected; decrease (0.07) if
                                    # phantom rings remain.
HOUGH_RADIUS_TOLERANCE = 0.40       # REJECT abnormally large / small rings: for fairly uniform droplets the radius
                                    # cannot stray far from the median. Reject any ring whose radius falls outside
                                    # [(1-tol), (1+tol)] x median. 0.40 = allow +/-40%.
                                    # increase (0.6) if your droplets vary a lot in size; decrease (0.25) to be stricter.

# --- Split touching droplets (only used when DETECTION_METHOD = "watershed") ---
USE_WATERSHED = True                # True: separate touching / overlapping droplets with watershed.
MIN_DISTANCE_PEAKS = 20             # minimum distance (px) between two droplet centres when splitting.
                                    # small / closely packed droplets -> decrease; large droplets -> increase.

# --- Circle drawing ---
DRAW_FILLED = False                 # False = draw circle outlines (the droplet underneath stays visible for checking).
                                    # True  = draw semi-transparent filled circles.
CIRCLE_COLOR = (0, 220, 0)          # circle colour (B, G, R) - default green.
CIRCLE_THICKNESS = 2                # circle outline thickness.
SHOW_INDEX = True                   # print the droplet index number on the image.

# --- Histogram ---
N_BINS = 12                         # number of histogram bars.
COMBINED_HISTOGRAM = True           # True = pool droplets from ALL images into a single combined histogram.

# Output folder. None -> defaults to the folder that contains the input images.
OUTPUT_DIR = "output"


# ============================================================
#  STEP 0: COLLECT THE LIST OF IMAGES
# ============================================================
def collect_image_paths():
    """Return the list of images to analyse, printing clear diagnostics about how many
    images were found and which files were skipped (so it is easy to see why an image is
    missing). Old result files are skipped automatically."""
    if IMAGE_PATHS:
        return list(IMAGE_PATHS)

    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Folder not found:\n  {INPUT_DIR}")
        print("      -> Check INPUT_DIR: correct folder name, correct path separators,")
        print("         and that the folder really contains images.")
        return []

    # collect candidates (optionally including sub-folders)
    candidates = []
    if SEARCH_SUBFOLDERS:
        for root, _, files in os.walk(INPUT_DIR):
            for nm in files:
                candidates.append(os.path.join(root, nm))
    else:
        for nm in os.listdir(INPUT_DIR):
            candidates.append(os.path.join(INPUT_DIR, nm))

    images, skipped_output, skipped_other = [], [], []
    for full in sorted(candidates):
        if not os.path.isfile(full):
            continue
        name = os.path.basename(full)
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            skipped_other.append(name)
            continue
        if name.startswith(("detected_", "histogram", "circles_")):
            skipped_output.append(name)   # skip previously generated result files
            continue
        images.append(full)

    # print diagnostics
    print(f"[Scan] Total files: {len(candidates)} | Valid images: {len(images)}")
    if skipped_output:
        print(f"  - Skipped {len(skipped_output)} old result files (detected_/circles_/histogram).")
    if skipped_other:
        show = ", ".join(skipped_other[:6]) + ("..." if len(skipped_other) > 6 else "")
        print(f"  - Skipped {len(skipped_other)} non-image files (unexpected extension): {show}")
        print("    (if one of these is actually your image, add its extension to IMAGE_EXTENSIONS)")
    for p in images:
        print(f"    - {os.path.basename(p)}")
    return images


# ============================================================
#  STEP 1: SCALE CALIBRATION (SCALE BAR)
# ============================================================
def detect_scale_bar(gray_img, search_region=(0.6, 1.0, 0.0, 0.5), min_len=15):
    """Auto-detect the length (px) of a white scale bar: find the longest horizontal run of
    near-white pixels (>200) in the lower-left corner of the image. Return the length in px,
    or None if nothing long enough is found."""
    h, w = gray_img.shape
    y0, y1 = int(h * search_region[0]), int(h * search_region[1])
    x0, x1 = int(w * search_region[2]), int(w * search_region[3])
    region = gray_img[y0:y1, x0:x1]

    best_len = 0
    for row in region:
        white = row > 200
        run = 0
        for v in white:
            run = run + 1 if v else 0
            best_len = max(best_len, run)
    return best_len if best_len >= min_len else None


def load_and_calibrate(image_path, forced_scale_px=None, forced_um_per_px=None):
    """Read an image and compute its micron-per-pixel ratio.
    - If forced_um_per_px is given, use it directly (highest priority).
    - If forced_scale_px is given (a shared scale bar), use that value.
    - Otherwise auto-detect the scale bar on this image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if forced_um_per_px is not None:
        um_per_px = forced_um_per_px
        print(f"  [Calibration] Using the directly entered ratio: 1 pixel = {um_per_px:.4f} um")
        return img, gray, um_per_px

    if forced_scale_px is not None:
        scale_px = forced_scale_px
        print(f"  [Calibration] Using the shared scale bar: {scale_px} px = {SCALE_BAR_UM} um")
    else:
        scale_px = detect_scale_bar(gray)
        if scale_px is None:
            raise ValueError(
                "Could not auto-detect a scale bar on this image. "
                "Set SHARED_SCALE_BAR=True to borrow a scale bar from another image, "
                "or set SCALE_BAR_PX / SCALE_UM_PER_PX manually."
            )
        print(f"  [Calibration] Auto-detected scale bar: {scale_px} px = {SCALE_BAR_UM} um")

    um_per_px = SCALE_BAR_UM / scale_px
    print(f"  [Calibration] -> 1 pixel = {um_per_px:.4f} um")
    return img, gray, um_per_px


def determine_shared_scale(paths):
    """Scan the images and return (scale_px, image_name) for the FIRST image where a scale bar
    is detected. Use this when every image shares the same magnification: one image with a clear
    scale bar is enough for the whole batch. Returns (None, None) if no image yields one."""
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale_px = detect_scale_bar(gray)
        if scale_px is not None:
            return scale_px, os.path.basename(path)
    return None, None


def resolve_scale_image(ref):
    """Locate the image that contains the scale bar. 'ref' may be a full path or just a file name.
    Search order: (1) ref as an exact path -> (2) ref joined onto INPUT_DIR ->
    (3) search for the file name inside INPUT_DIR and all its sub-folders. Returns None if not found."""
    if os.path.isfile(ref):
        return ref
    cand = os.path.join(INPUT_DIR, ref)
    if os.path.isfile(cand):
        return cand
    target = os.path.basename(ref)
    if os.path.isdir(INPUT_DIR):
        for root, _, files in os.walk(INPUT_DIR):
            if target in files:
                return os.path.join(root, target)
    return None


def scale_px_from_reference_image(ref):
    """Read the reference scale-bar image and detect the scale-bar length (px).
    Return scale_px, or raise a clear error if the image cannot be found / detected."""
    ref_path = resolve_scale_image(ref)
    if ref_path is None:
        raise FileNotFoundError(
            f"Scale-bar image not found: {ref}\n"
            f"  -> Check SCALE_BAR_IMAGE (correct file name / path)."
        )
    img = cv2.imread(ref_path)
    if img is None:
        raise ValueError(f"Could not read the scale-bar image: {ref_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_px = detect_scale_bar(gray)
    if scale_px is None:
        raise ValueError(
            f"No scale bar could be detected in '{os.path.basename(ref_path)}'.\n"
            f"  -> Check that the image has a white scale bar in the lower-left corner, "
            f"or measure it by hand and set SCALE_BAR_PX."
        )
    return scale_px, os.path.basename(ref_path)


# ============================================================
#  STEP 2: DETECT + SPLIT DROPLETS
# ============================================================
def segment_particles(img):
    """Build a binary mask (droplet = white, background = black).

    The pipeline is tuned for magenta droplets on a grey background:
    - Otsu threshold on the Saturation channel (droplets are highly saturated, the grey
      background is not).
    - Morphology with an ELLIPSE (round) kernel to clean up without distorting round shapes.
    - FILL HOLES inside each droplet - important for droplets with a dark centre / rim,
      otherwise they end up punctured and are measured incorrectly."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)  # bridge gaps, patch openings
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)   # remove speckle noise
    mask = ndi.binary_fill_holes(mask > 0)                          # fill holes inside droplets
    return mask


def label_particles(mask):
    """Label the droplets. With watershed on -> split touching droplets; off -> label the
    connected components directly."""
    if not USE_WATERSHED:
        labels, _ = ndi.label(mask)
        return labels

    dist = ndi.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=MIN_DISTANCE_PEAKS, labels=mask)
    markers = np.zeros(dist.shape, dtype=int)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    labels = watershed(-dist, markers, mask=mask)
    return labels


# ============================================================
#  STEP 3: FIT A CIRCLE TO EACH DROPLET
# ============================================================
def measure_particles(img, labels, um_per_px):
    """For each droplet: take its contour, compute the area, then convert to an EQUIVALENT
    CIRCULAR DIAMETER (ECD): diameter = 2 * sqrt(area / pi). The circle centre is the droplet
    centroid. This is the standard way to size particles, and the drawn circle matches the
    reported number."""
    h, w = labels.shape
    results = []
    for lab in range(1, labels.max() + 1):
        region = labels == lab
        area_px = int(region.sum())
        if area_px < MIN_PARTICLE_AREA_PX:
            continue

        mean_green = img[region][:, 1].mean()
        if mean_green > MAX_GREEN_CHANNEL:
            continue  # off-colour object, not a droplet

        ys, xs = np.where(region)
        touches_edge = ys.min() == 0 or ys.max() == h - 1 or xs.min() == 0 or xs.max() == w - 1
        if EXCLUDE_EDGE_PARTICLES and touches_edge:
            continue

        region_u8 = (region.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(cnt)
        if contour_area < MIN_PARTICLE_AREA_PX:
            continue

        # Equivalent circular diameter (ECD).
        r_eq = np.sqrt(contour_area / np.pi)
        diameter_px = 2.0 * r_eq
        cx, cy = xs.mean(), ys.mean()

        # For reference: minimum enclosing circle (used to gauge circularity if needed).
        (mx, my), r_min = cv2.minEnclosingCircle(cnt)
        circularity = contour_area / (np.pi * r_min * r_min) if r_min > 0 else 0  # ~1 means nicely round

        results.append({
            "cx": cx, "cy": cy,
            "radius_px": r_eq,
            "diameter_px": diameter_px,
            "diameter_um": diameter_px * um_per_px,
            "circularity": circularity,
        })
    return results


# ---------- DETECTION WITH THE HOUGH CIRCLE TRANSFORM (for large, dense, overlapping droplets) ----------
def estimate_radius_range(img):
    """Auto-estimate the droplet radius range (px) with one coarse, wide-band Hough pass, taken
    around the median radius. This lets the parameters adapt to large / small images without
    manual tuning."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(gray, 7)
    H, W = gray.shape
    rmin_wide = max(8, int(min(H, W) * 0.02))
    rmax_wide = int(min(H, W) * 0.30)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(20, rmin_wide),
                               param1=120, param2=45, minRadius=rmin_wide, maxRadius=rmax_wide)
    if circles is None or len(circles[0]) < 3:
        return rmin_wide, rmax_wide
    med = float(np.median(circles[0][:, 2]))
    return max(8, int(med * 0.6)), int(med * 1.4)


def _edge_support(gray, x, y, r, dark_thresh, tol=25, n=90):
    """Fraction of points on the circumference that have a dark rim (brightness < dark_thresh)
    within the radius band [r-tol, r+tol]. Measures how well the circle hugs the droplet rim
    (1 = perfect fit)."""
    H, W = gray.shape
    hit = 0
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        ca, sa = np.cos(a), np.sin(a)
        for dr in range(-tol, tol + 1, 3):
            xi = int(x + (r + dr) * ca); yi = int(y + (r + dr) * sa)
            if 0 <= xi < W and 0 <= yi < H and gray[yi, xi] < dark_thresh:
                hit += 1
                break
    return hit / n


def _inner_dark_frac(gray, x, y, r, dark_thresh, frac=0.45):
    """Fraction of dark pixels (dark rim) inside the CORE region (radius r*frac around the centre).
    A real droplet -> clean magenta core -> small value (usually <9%). A phantom ring spanning a
    gap / several droplets -> a dark rim crosses its core -> large value (>15%). A 45% core is
    inner enough that the dark crescent at the rim of a real droplet does NOT leak in, so real
    droplets are not rejected by mistake."""
    H, W = gray.shape
    rr = int(r * frac)
    if rr < 3:
        return 0.0
    y0, y1 = max(0, y - rr), min(H, y + rr)
    x0, x1 = max(0, x - rr), min(W, x + rr)
    patch = gray[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= rr * rr
    vals = patch[inside]
    return float((vals < dark_thresh).mean()) if vals.size else 0.0


def _auto_dark_thresh(img):
    """Auto-compute the 'dark rim' threshold for an image: 0.7 x the median brightness of the
    magenta interior (highly saturated pixels). This value always sits BETWEEN the dark rim and
    the bright core, so it works for both bright and dark magenta images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, ms = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    interior = gray[ms > 0]
    if interior.size == 0:
        return 70
    return int(0.7 * float(np.median(interior)))


def detect_particles_hough(img, um_per_px):
    """Detect droplets with the Hough Circle Transform, then FILTER:
    1) score each ring on how well it hugs the rim, and reject poor fits (low score);
    2) reject PHANTOM RINGS spanning a gap / several droplets (a dark rim crosses the core);
    3) reject duplicate rings (centres too close = the same droplet).
    Returns a list of results in the same format as measure_particles."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(gray, 7)
    H, W = gray.shape

    dark_thresh = HOUGH_DARK_THRESH if HOUGH_DARK_THRESH is not None else _auto_dark_thresh(img)

    rlo, rhi = HOUGH_RADIUS_PX
    if rlo is None or rhi is None:
        rlo, rhi = estimate_radius_range(img)
    min_dist = max(20, int(rlo * 1.2))

    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
                               param1=120, param2=HOUGH_PARAM2,
                               minRadius=int(rlo), maxRadius=int(rhi))
    print(f"  [Hough] Radius range [{int(rlo)},{int(rhi)}]px, rim threshold {dark_thresh} | "
          f"{0 if circles is None else len(circles[0])} raw rings")
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)

    # score rim support + reject bad fits AND phantom rings (dark rim crossing the core)
    scored = []
    n_phantom = 0
    for x, y, r in circles:
        s = _edge_support(gray, x, y, r, dark_thresh)
        if s < HOUGH_MIN_EDGE_SUPPORT:
            continue
        if _inner_dark_frac(gray, x, y, r, dark_thresh) > HOUGH_MAX_INNER_DARK:
            n_phantom += 1
            continue  # phantom ring spanning a gap / several droplets -> reject
        scored.append((s, x, y, r))
    scored.sort(reverse=True)
    if n_phantom:
        print(f"  [Hough] Rejected {n_phantom} phantom rings (spanning a gap / droplets, core not clean)")

    # remove duplicates: drop any ring whose centre is too close to an already-kept higher-scored ring
    kept = []
    for s, x, y, r in scored:
        if any(np.hypot(x - xk, y - yk) < 0.6 * min(r, rk) for _, xk, yk, rk in kept):
            continue
        kept.append((s, x, y, r))

    # REJECT ABNORMAL RADII: for fairly uniform droplets, rings far from the median radius are
    # almost certainly phantom (Hough merged the outer edge of a whole cluster into one big ring).
    if HOUGH_RADIUS_TOLERANCE is not None and len(kept) >= 4:
        med_r = float(np.median([r for _, _, _, r in kept]))
        lo_r = med_r / (1.0 + HOUGH_RADIUS_TOLERANCE)
        hi_r = med_r * (1.0 + HOUGH_RADIUS_TOLERANCE)
        before = len(kept)
        kept = [(s, x, y, r) for (s, x, y, r) in kept if lo_r <= r <= hi_r]
        n_out = before - len(kept)
        if n_out:
            print(f"  [Hough] Rejected {n_out} rings with an abnormal radius "
                  f"(outside [{lo_r:.0f},{hi_r:.0f}]px vs median {med_r:.0f}px)")

    results = []
    for s, x, y, r in kept:
        if EXCLUDE_EDGE_PARTICLES and (x - r < 0 or y - r < 0 or x + r >= W or y + r >= H):
            continue  # ring sticks out past the border -> clipped droplet (if set to exclude)
        results.append({
            "cx": float(x), "cy": float(y),
            "radius_px": float(r),
            "diameter_px": 2.0 * r,
            "diameter_um": 2.0 * r * um_per_px,
            "circularity": float(s),   # for Hough: the rim-support score (fit quality), 1 = perfect
        })
    return results


def detect_particles(img, um_per_px):
    """Dispatcher: choose the droplet-detection method according to DETECTION_METHOD."""
    if DETECTION_METHOD == "hough":
        return detect_particles_hough(img, um_per_px)
    # default: watershed + equivalent-circle fit
    mask = segment_particles(img)
    labels = label_particles(mask)
    return measure_particles(img, labels, um_per_px)


# ============================================================
#  STEP 4: EXPORT RESULTS
# ============================================================
def draw_circles(img, results, out_path):
    """Draw the fitted CIRCLES on top of the original image (green outline by default) + index numbers."""
    vis = img.copy()
    overlay = img.copy()
    for i, r in enumerate(results, 1):
        center = (int(round(r["cx"])), int(round(r["cy"])))
        radius = int(round(r["radius_px"]))
        if DRAW_FILLED:
            cv2.circle(overlay, center, radius, CIRCLE_COLOR, -1)
        else:
            cv2.circle(vis, center, radius, CIRCLE_COLOR, CIRCLE_THICKNESS)
    if DRAW_FILLED:
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        for i, r in enumerate(results, 1):
            center = (int(round(r["cx"])), int(round(r["cy"])))
            cv2.circle(vis, center, int(round(r["radius_px"])), CIRCLE_COLOR, 1)

    if SHOW_INDEX:
        for i, r in enumerate(results, 1):
            center = (int(round(r["cx"])), int(round(r["cy"])))
            cv2.putText(vis, str(i), (center[0] - 7, center[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)
    print(f"  [Export] Circle image: {os.path.basename(out_path)}")


def plot_histogram(diameters_um, out_path, title="Particle Size Distribution"):
    diameters_um = np.array(diameters_um, dtype=float)
    mean = diameters_um.mean()
    std = diameters_um.std()
    cv_pct = 100 * std / mean if mean > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    counts, bins, _ = ax.hist(
        diameters_um, bins=N_BINS, edgecolor="black", color="#f472b6", alpha=0.85
    )
    if std > 0 and len(diameters_um) > 1:
        x = np.linspace(diameters_um.min() * 0.7, diameters_um.max() * 1.3, 300)
        bin_width = bins[1] - bins[0]
        gaussian = (len(diameters_um) * bin_width / (std * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((x - mean) / std) ** 2))
        ax.plot(x, gaussian, color="#1f2937", linewidth=2)
    ax.axvline(mean, color="crimson", linestyle="--", linewidth=1, label=f"Mean = {mean:.1f} um")

    ax.set_xlabel("Diameter (um)", fontsize=12)
    ax.set_ylabel("n (count)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.97, 0.95,
            f"n = {len(diameters_um)}\nMean = {mean:.1f} um\nStd = {std:.1f} um\nCV = {cv_pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"  [Export] Histogram: {os.path.basename(out_path)}")
    plt.close(fig)
    return mean, std, cv_pct


def export_csv_combined(all_rows, out_path):
    """One combined CSV for every image, with an 'image' column. utf-8-sig so Excel reads the
    um symbol correctly."""
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "particle_id", "diameter_um", "diameter_px",
                         "circularity", "center_x_px", "center_y_px"])
        counters = {}
        for img_name, r in all_rows:
            counters[img_name] = counters.get(img_name, 0) + 1
            writer.writerow([img_name, counters[img_name],
                             round(r["diameter_um"], 2), round(r["diameter_px"], 2),
                             round(r["circularity"], 3),
                             round(r["cx"], 1), round(r["cy"], 1)])
    print(f"[Export] Combined data table: {os.path.basename(out_path)}")


# ============================================================
#  MAIN PROGRAM - PROCESS MULTIPLE IMAGES
# ============================================================
def main():
    paths = collect_image_paths()
    if not paths:
        where = "IMAGE_PATHS" if IMAGE_PATHS else f"folder:\n  {INPUT_DIR}"
        print(f"[Error] No images found in {where}")
        return
    print(f"[Start] Found {len(paths)} image(s) to analyse.\n")

    # --- Determine the shared scale (applied to EVERY image) ---
    # Priority: direct um/px > manual SCALE_BAR_PX > reference IMAGE > shared auto-detect > per-image.
    shared_scale_px = None
    if SCALE_UM_PER_PX is not None:
        print(f"[Scale] Using the directly entered SCALE_UM_PER_PX = {SCALE_UM_PER_PX} um/px for every image.\n")
        # go straight into the loop; each image uses forced_um_per_px
    elif SCALE_BAR_PX is not None:
        shared_scale_px = SCALE_BAR_PX
        print(f"[Scale bar] Using the manual SCALE_BAR_PX = {SCALE_BAR_PX} px "
              f"(= {SCALE_BAR_UM} um) for every image.\n")
    elif SCALE_BAR_IMAGE:
        try:
            shared_scale_px, src = scale_px_from_reference_image(SCALE_BAR_IMAGE)
        except (FileNotFoundError, ValueError) as e:
            print(f"[Scale bar] {e}")
            return
        print(f"[Scale bar] Detected {shared_scale_px} px from the REFERENCE image '{src}', "
              f"applied to all {len(paths)} images (= {SCALE_BAR_UM} um).\n")
    elif SHARED_SCALE_BAR:
        shared_scale_px, src = determine_shared_scale(paths)
        if shared_scale_px is None:
            print("[Scale bar] Could not detect a scale bar in ANY image.\n"
                  "  -> Set SCALE_UM_PER_PX (um/px ratio), specify SCALE_BAR_IMAGE, "
                  "or measure by hand and set SCALE_BAR_PX at the top of the file.")
            return
        print(f"[Scale bar] Auto-detected {shared_scale_px} px from image '{src}', "
              f"applied to all {len(paths)} images (= {SCALE_BAR_UM} um).\n")
    # if everything is off -> shared_scale_px = None -> each image detects its own scale bar

    out_dir = OUTPUT_DIR or os.path.dirname(os.path.abspath(paths[0]))
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    all_diameters = []
    per_image_summary = []

    for idx, path in enumerate(paths, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"--- [{idx}/{len(paths)}] {os.path.basename(path)} ---")
        try:
            img, gray, um_per_px = load_and_calibrate(
                path, forced_scale_px=shared_scale_px, forced_um_per_px=SCALE_UM_PER_PX)
            results = detect_particles(img, um_per_px)
            print(f"  Detected {len(results)} valid droplet(s).")

            if not results:
                print("  [Warning] No droplet measured (try lowering MIN_PARTICLE_AREA_PX "
                      "or re-checking the image).\n")
                continue

            draw_circles(img, results, os.path.join(out_dir, f"circles_{name}.png"))

            di_um = [r["diameter_um"] for r in results]
            all_diameters.extend(di_um)
            for r in results:
                all_rows.append((os.path.basename(path), r))
            per_image_summary.append((os.path.basename(path), len(results),
                                      float(np.mean(di_um)), float(np.std(di_um))))

            if not COMBINED_HISTOGRAM:
                plot_histogram(di_um, os.path.join(out_dir, f"histogram_{name}.png"),
                               title=f"Size Distribution - {name}")
        except Exception as e:
            # one failing image must not stop the whole batch -> report and continue
            print(f"  [Error on this image, skipped so the batch can continue] "
                  f"{type(e).__name__}: {e}")
        print()

    if not all_diameters:
        print("[Done] No droplet was measured across all images.")
        return

    export_csv_combined(all_rows, os.path.join(out_dir, "particle_diameters_all.csv"))

    if COMBINED_HISTOGRAM:
        mean, std, cv_pct = plot_histogram(
            all_diameters, os.path.join(out_dir, "histogram_combined.png"),
            title=f"Particle Size Distribution (pooled, {len(per_image_summary)} images)")
    else:
        mean = float(np.mean(all_diameters)); std = float(np.std(all_diameters))
        cv_pct = 100 * std / mean if mean > 0 else 0.0

    print("\n===== PER-IMAGE SUMMARY =====")
    print(f"{'Image':<28}{'Count':>8}{'Mean(um)':>10}{'SD(um)':>10}")
    for nm, n, m, s in per_image_summary:
        print(f"{nm:<28}{n:>8}{m:>10.1f}{s:>10.1f}")

    print("\n===== OVERALL SUMMARY =====")
    print(f"Images analysed: {len(per_image_summary)}/{len(paths)}")
    print(f"Total droplets measured: {len(all_diameters)}")
    print(f"Mean diameter: {mean:.1f} +/- {std:.1f} um")
    print(f"Coefficient of variation (CV): {cv_pct:.1f}% "
          f"({'fairly uniform (monodisperse)' if cv_pct < 10 else 'noticeable size spread'})")


if __name__ == "__main__":
    main()
