"""
Chinese Painting 3D Digitization Evaluation Metrics
Each metric returns a score in [0, 1], higher = more similar to reference.

Dependencies:
    pip install opencv-python scikit-image lpips torch torchvision pillow numpy
"""

import ssl
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as _ssim
import torch
from torchvision import transforms, models
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context


# ─── PROFESSIONAL METRICS ────────────────────────────────────────────────────

def compute_lpips(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """
    Perceptual similarity via mean cosine similarity of VGG11 multi-layer features.
    Normalized to [-1,1] range. Extracts features after each ReLU and averages cosine similarity.
    """
    vgg = models.vgg11(pretrained=True).features.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ref_t = to_tensor(Image.fromarray(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    tst_t = to_tensor(Image.fromarray(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))).unsqueeze(0)

    sims = []
    x_r, x_t = ref_t, tst_t
    with torch.no_grad():
        for layer in vgg:
            x_r = layer(x_r)
            x_t = layer(x_t)
            if isinstance(layer, torch.nn.ReLU):
                sim = torch.nn.functional.cosine_similarity(
                    x_r.flatten().unsqueeze(0), x_t.flatten().unsqueeze(0)
                ).item()
                sims.append(sim)
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def compute_ssim(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """
    SSIM on color image, 5×5 window.
    Requires pixel-level alignment; scores reflect structural similarity directly.
    """
    ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    tst_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
    score = _ssim(ref_rgb, tst_rgb, win_size=5, channel_axis=2, data_range=255)
    return float(np.clip(score, 0.0, 1.0))


# ─── COLOR METRICS ────────────────────────────────────────────────────────────

def compute_color_fidelity(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """
    Color fidelity via mean absolute difference per pixel in BGR space.
    score = 1 - mean_diff / 255.
    Requires pixel-level alignment for meaningful results.
    """
    diff = np.abs(img_ref.astype(np.float32) - img_test.astype(np.float32))
    return float(np.clip(1.0 - diff.mean() / 255.0, 0.0, 1.0))


# ─── STYLE METRICS ────────────────────────────────────────────────────────────

def compute_brushstroke_texture(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """
    Brushstroke texture via LBP histogram Bhattacharyya distance.
    Parameters: radius=3, n_points=24 (uniform LBP).
    score = 1 - bhattacharyya_distance.
    """
    radius, n_points = 3, 24
    ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    tst_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

    lbp_ref = local_binary_pattern(ref_gray, n_points, radius, method="uniform")
    lbp_tst = local_binary_pattern(tst_gray, n_points, radius, method="uniform")

    n_bins = n_points + 2
    hist_ref, _ = np.histogram(lbp_ref.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist_tst, _ = np.histogram(lbp_tst.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return float(np.sum(np.minimum(hist_ref, hist_tst)))


# ─── TEXTURE METRICS ─────────────────────────────────────────────────────────

def compute_surface_texture(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """
    Surface texture similarity based on variance of Laplacian responses.
    score = min(var_ref, var_tst) / max(var_ref, var_tst).
    """
    ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tst_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY).astype(np.float32)

    var_ref = float(np.var(cv2.Laplacian(ref_gray, cv2.CV_32F)))
    var_tst = float(np.var(cv2.Laplacian(tst_gray, cv2.CV_32F)))

    denom = max(var_ref, var_tst)
    if denom < 1e-10:
        return 1.0
    return float(np.clip(min(var_ref, var_tst) / denom, 0.0, 1.0))


def compute_glossiness(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """
    Glossiness: Bhattacharyya similarity of HSV-V histograms (0.55 weight)
              + highlight-pixel ratio similarity (0.45 weight).
    """
    ref_hsv = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
    tst_hsv = cv2.cvtColor(img_test, cv2.COLOR_BGR2HSV)

    hist_ref = cv2.calcHist([ref_hsv], [2], None, [256], [0, 256])
    hist_tst = cv2.calcHist([tst_hsv], [2], None, [256], [0, 256])
    cv2.normalize(hist_ref, hist_ref)
    cv2.normalize(hist_tst, hist_tst)
    bhatt_dist = cv2.compareHist(hist_ref, hist_tst, cv2.HISTCMP_BHATTACHARYYA)
    hist_score = float(np.exp(-0.01 * bhatt_dist))

    v_ref = ref_hsv[:, :, 2].astype(np.float32)
    v_tst = tst_hsv[:, :, 2].astype(np.float32)
    hl_ref = float(np.mean(v_ref > 220))
    hl_tst = float(np.mean(v_tst > 220))
    highlight_score = float(np.clip(1.0 - abs(hl_ref - hl_tst), 0.0, 1.0))

    return float(np.clip(0.55 * hist_score + 0.45 * highlight_score, 0.0, 1.0))


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def evaluate(ref_path: str, test_path: str) -> dict:
    img_ref = cv2.imread(ref_path)
    img_tst = cv2.imread(test_path)
    assert img_ref is not None, f"Cannot load reference image: {ref_path}"
    assert img_tst is not None, f"Cannot load test image: {test_path}"

    if img_ref.shape != img_tst.shape:
        img_tst = cv2.resize(img_tst, (img_ref.shape[1], img_ref.shape[0]))

    return {
        "lpips":           compute_lpips(img_ref, img_tst),
        "ssim":            compute_ssim(img_ref, img_tst),
        "color_fidelity":  compute_color_fidelity(img_ref, img_tst),
        "brushstroke":     compute_brushstroke_texture(img_ref, img_tst),
        "surface_texture": compute_surface_texture(img_ref, img_tst),
        "glossiness":      compute_glossiness(img_ref, img_tst),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python metrics.py <reference_image> <test_image>")
        sys.exit(1)

    scores = evaluate(sys.argv[1], sys.argv[2])

    print("\n===== PROFESSIONAL METRICS =====")
    print(f"LPIPS: {scores['lpips']:.2f}")
    print(f"SSIM:  {scores['ssim']:.2f}")

    print("\n===== COLOR METRICS =====")
    print(f"color fidelity: {scores['color_fidelity']:.2f}")

    print("\n===== STYLE METRICS =====")
    print(f"brushstroke: {scores['brushstroke']:.2f}")

    print("\n===== TEXTURE METRICS =====")
    print(f"surface texture:  {scores['surface_texture']:.2f}")
    print(f"gloss consistency: {scores['glossiness']:.2f}")
