"""
Robust captcha solver for noisy, stylized captchas.
Two-pronged approach:
  - multi-variant whole-image OCR (ensemble by confidence)
  - segmentation-based OCR (find blobs, split wide blobs, OCR each char)
Postprocessing:
  - remove non-whitelisted chars
  - apply small confusion-correction map (tuneable)

Edit IMAGE_PATH and EXPECTED_LEN as needed.
"""

import cv2
import numpy as np
import easyocr
import os
from statistics import mean

# ---------- USER CONFIG ----------
IMAGE_PATH = r"C:\Users\Lenovo\Desktop\CAPTCHA PROJECT\captcha.png"   # path to your image
EXPECTED_LEN = None         # set to 5 if you know captcha length (helps)
USE_GPU_IF_AVAILABLE = False  # set True if you installed GPU PyTorch & have GPU
DEBUG_SAVE = True           # saves intermediate images for inspection
OUT_DIR = "debug_out"
# ---------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# init reader (auto-detect GPU if requested and available)
try:
    import torch
    gpu_ok = torch.cuda.is_available() and USE_GPU_IF_AVAILABLE
except Exception:
    gpu_ok = False

reader = easyocr.Reader(['en'], gpu=gpu_ok)

# whitelist: allowed characters in captcha (modify if your captcha includes lowercase)
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
# small confusion map to correct visually-similar misreads; tune as needed
CONFUSION_MAP = {
   # "A": "4", "a": "4",
    "O": "0", "o": "0",
    "I": "1", "l": "1", "i": "1",
    "Z": "2", "S": "5", "B": "8",
    "g": "9",
}

def apply_confusion_map(s):
    return "".join(CONFUSION_MAP.get(ch, ch) for ch in s)

def keep_whitelist(s):
    return "".join(ch for ch in s if ch in WHITELIST)

# ---------- Preprocessing variants ----------
def make_variants(bgr):
    """
    Return list of variants as BGR or grayscale images suitable for easyocr and segmentation.
    """
    variants = []
    h, w = bgr.shape[:2]

    # basic gray and CLAHE
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(clahe)

    # bilateral (edge-preserving) then adaptive threshold
    bi = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    variants.append(bi)

    # top-hat (emphasize bright on dark)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    variants.append(tophat)

    # background subtraction (large open)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
    sub = cv2.subtract(gray, bg)
    variants.append(sub)

    # inverted (sometimes strokes are lighter)
    variants.append(cv2.bitwise_not(gray))

    # adaptive threshold versions - add binary results too
    for v in [clahe, bi, tophat, sub]:
        th = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)
        variants.append(th)

    # morphological clean variants
    cleaned = []
    for v in variants:
        if len(v.shape) == 2:
            kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            open_v = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel_small, iterations=1)
            close_v = cv2.morphologyEx(open_v, cv2.MORPH_CLOSE, kernel_small, iterations=1)
            cleaned.append(close_v)
    variants.extend(cleaned)

    # scale-up tiny images (OCR likes larger text)
    scaled = []
    for v in variants:
        h, w = v.shape[:2]
        factor = 2  # upscale factor
        scaled_v = cv2.resize(v, (w*factor, h*factor), interpolation=cv2.INTER_CUBIC)
        scaled.append(scaled_v)
    variants.extend(scaled)

    # deduplicate by shape
    uniq = []
    seen = set()
    for v in variants:
        key = (v.shape, v.dtype, v.tobytes()[:32])
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    return uniq

# ---------- Whole-image OCR (ensemble) ----------
def ocr_whole_image_variants(bgr):
    variants = make_variants(bgr)
    best_text = ""
    best_score = -1.0
    debug_idx = 0
    for var in variants:
        # EasyOCR accepts grayscale or color; pass as is
        try:
            res = reader.readtext(var, detail=1, paragraph=False)
        except Exception:
            res = []
        if not res:
            continue
        texts = [r[1] for r in res]
        confs = [r[2] for r in res if r[2] is not None]
        avg_conf = mean(confs) if confs else 0.0
        joined = "".join(texts).strip()
        # postprocess joined text
        cleaned = keep_whitelist(joined)
        cleaned = apply_confusion_map(cleaned)
        # prefer correct length if expected len provided
        score = avg_conf
        if EXPECTED_LEN is not None and len(cleaned) == EXPECTED_LEN:
            score += 0.5  # boost if meets expected length
        if score > best_score and cleaned:
            best_score = score
            best_text = cleaned
        # optional debug save
        if DEBUG_SAVE:
            path = os.path.join(OUT_DIR, f"variant_{debug_idx}_ocr_{cleaned}_conf_{avg_conf:.2f}.png")
            cv2.imwrite(path, var)
        debug_idx += 1
    return best_text, best_score

# ---------- Segmentation-based OCR ----------
def segment_and_ocr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Pre-clean: CLAHE + median denoise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    den = cv2.medianBlur(clahe, 3)

    # background removal
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    bg = cv2.morphologyEx(den, cv2.MORPH_OPEN, k)
    sub = cv2.subtract(den, bg)

    # adaptive threshold
    th = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 3)

    # remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    if DEBUG_SAVE:
        cv2.imwrite(os.path.join(OUT_DIR, r"C:\Users\Lenovo\Desktop\CAPTCHA PROJECT\seg_clean.png"), clean)

    # connected components to get candidate blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    boxes = []
    img_h, img_w = clean.shape[:2]
    areas = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        areas.append(area)
    if not areas:
        return "", 0.0  # nothing found

    median_area = np.median([stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)])
    # collect boxes filtered by reasonable size
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # filter tiny specks and overly large background components
        if area < max(10, 0.01 * median_area): 
            continue
        if h < 8 or w < 4:
            continue
        # ignore components that touch image border heavily (maybe background)
        if x <= 1 and w > img_w * 0.9:
            continue
        boxes.append((x, y, w, h, area))

    if not boxes:
        return "", 0.0

    # Sort boxes by x (left-to-right)
    boxes = sorted(boxes, key=lambda b: b[0])

    # merge boxes that are very close horizontally (likely parts of same char)
    merged = []
    if boxes:
        cur = boxes[0]
        for nx in boxes[1:]:
            gap = nx[0] - (cur[0] + cur[2])
            # if negative overlap or small gap, merge
            if gap < max(6, 0.05 * img_w):
                # merge
                x1 = min(cur[0], nx[0])
                y1 = min(cur[1], nx[1])
                x2 = max(cur[0] + cur[2], nx[0] + nx[2])
                y2 = max(cur[1] + cur[3], nx[1] + nx[3])
                cur = (x1, y1, x2 - x1, y2 - y1, cur[4] + nx[4])
            else:
                merged.append(cur)
                cur = nx
        merged.append(cur)
    else:
        merged = boxes

    # If any merged box is very wide (likely contains 2 or more chars), attempt vertical split
    char_crops = []
    widths = [b[2] for b in merged]
    median_w = max(1, int(np.median(widths)))
    for (x, y, w, h, area) in merged:
        # padding
        pad_x = int(0.08 * w) + 2
        pad_y = int(0.08 * h) + 2
        x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
        x1 = min(img_w, x + w + pad_x); y1 = min(img_h, y + h + pad_y)
        roi = clean[y0:y1, x0:x1]
        # if wide relative to median width, try splitting by vertical projection valleys
        if w > 1.6 * median_w and w > 30:
            cols_sum = roi.sum(axis=0)
            # normalize and find valleys
            cols_norm = (cols_sum - cols_sum.min()) / (cols_sum.max() - cols_sum.min() + 1e-9)
            # find indices where cols_norm is small (valley)
            valley_idxs = np.where(cols_norm < 0.25)[0]
            # split at largest gaps between valley clusters to get 2 or 3 pieces
            if valley_idxs.size > 0:
                # find candidate cut positions by selecting valley idxs that are local minima
                cuts = []
                # group contiguous valley indices
                groups = np.split(valley_idxs, np.where(np.diff(valley_idxs) != 1)[0] + 1)
                for g in groups:
                    if g.size > 0:
                        cuts.append(int(np.median(g)))
                # choose up to 2 cuts (for 2-3 chars)
                cuts = sorted(cuts)[:2]
                last = 0
                for c in cuts:
                    x_cut = int(c)
                    crop = roi[:, last:x_cut]
                    if crop.shape[1] > 3:
                        char_crops.append(((x0 + last, y0), crop))
                    last = x_cut
                # add remaining
                crop = roi[:, last:roi.shape[1]]
                if crop.shape[1] > 3:
                    char_crops.append(((x0 + last, y0), crop))
                continue
        # otherwise single crop
        char_crops.append(((x0, y0), roi))

    # Final OCR on each crop (pad and resize)
    ocr_results = []
    for (pos, crop) in char_crops:
        # invert back to dark on white if needed
        # convert to BGR for easyocr if single-channel
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        # upscale for better recognition
        h0, w0 = crop_rgb.shape[:2]
        scale = max(1, int(48 / min(h0, w0)))  # make smallest dimension ~48
        if scale > 1:
            crop_rgb = cv2.resize(crop_rgb, (w0*scale, h0*scale), interpolation=cv2.INTER_CUBIC)
        try:
            res = reader.readtext(crop_rgb, detail=1, paragraph=False)
        except Exception:
            res = []
        if res:
            # pick top result
            text = "".join([r[1] for r in res]).strip()
            confs = [r[2] for r in res if r[2] is not None]
            avg_conf = mean(confs) if confs else 0.0
        else:
            text = ""
            avg_conf = 0.0
        # clean text and map confusions
        text = keep_whitelist(text)
        text = apply_confusion_map(text)
        ocr_results.append((pos[0], text, avg_conf))
        if DEBUG_SAVE:
            # save crop for debug
            fname = os.path.join(OUT_DIR, f"crop_x{pos[0]}_t_{text}_c_{avg_conf:.2f}.png")
            cv2.imwrite(fname, crop)

    # sort results by x and join
    if not ocr_results:
        return "", 0.0
    ocr_results = sorted(ocr_results, key=lambda r: r[0])
    joined = "".join([r[1] for r in ocr_results])
    # average confidence
    avg_conf = mean([r[2] for r in ocr_results]) if ocr_results else 0.0
    # prefer result with expected len
    score = avg_conf
    if EXPECTED_LEN is not None and len(joined) == EXPECTED_LEN:
        score += 0.5
    return joined, score

# ---------- Main ----------
def solve(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # Whole-image ensemble
    whole_text, whole_score = ocr_whole_image_variants(bgr)
    print(f"[Whole-image] text='{whole_text}' score={whole_score:.3f}")

    # Segmentation
    seg_text, seg_score = segment_and_ocr(bgr)
    print(f"[Segmentation] text='{seg_text}' score={seg_score:.3f}")

    # Choose best
    final_text = ""
    final_score = -1.0
    if len(whole_text) > 0:
        final_text, final_score = whole_text, whole_score
    if len(seg_text) > 0 and seg_score > final_score:
        final_text, final_score = seg_text, seg_score

    # Final cleanup: remove stray spaces and ensure whitelist, optionally force expected length
    final_text = keep_whitelist(final_text)
    final_text = apply_confusion_map(final_text)

    print("------------------------------")
    print("FINAL:", final_text, f"(score={final_score:.3f})")
    print("------------------------------")
    # Save final preprocessed image for debugging if needed
    if DEBUG_SAVE:
        cv2.imwrite(os.path.join(OUT_DIR, r"C:\Users\Lenovo\Desktop\CAPTCHA PROJECT\input.png"), bgr)
    return final_text

if __name__ == "__main__":
    print("Processing:", IMAGE_PATH)
    out = solve(IMAGE_PATH)
    print("Done.")

