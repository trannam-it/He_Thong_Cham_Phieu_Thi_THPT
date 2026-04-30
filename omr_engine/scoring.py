# =============================================================================
# SCORING - Cac ham tinh diem "to" cua mot bubble tren anh da nan chinh.
# ---------------------------------------------------------------------------
# NANG CAP: Ho tro anh mo, to khong ro
#   - Dung 3 chi so ket hop: fill_ratio, darkness, gradient response.
#   - Adaptive thresholding cho tung bubble dua theo moi truong sang.
# =============================================================================

import cv2
import numpy as np


def _gamma_correct(gray, gamma=0.8):
    """Lam toi buble bang gamma < 1."""
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray, table)


def crop_bubble(gray, cx, cy, r, extra=1.15):
    """Cat vung vuong quanh bubble voi radius * extra."""
    H, W = gray.shape
    rr = max(4, int(round(r * extra)))
    x1 = max(0, int(round(cx)) - rr)
    y1 = max(0, int(round(cy)) - rr)
    x2 = min(W, int(round(cx)) + rr + 1)
    y2 = min(H, int(round(cy)) + rr + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return gray[y1:y2, x1:x2]


def bubble_fill_ratio(gray, cx, cy, r):
    """Tinh ty le pixel den trong LOI TRUNG TAM cua mot bubble.

    NANG CAP: Ket hop 3 chi so de chong anh mo/to khong ro:
      - fill_ratio : ty le pixel "den" theo nguong co dinh (140).
      - darkness   : 1 - mean_gray / 255 (bubble to dam cang cao).
      - local_adapt: ty le pixel DUOI MEAN CUC BO cua ROI rong.
                     Hieu qua khi anh bi mo hoac co bong.

    Empty bubble: ~0.00 - 0.15.
    Filled bubble: ~0.55 - 1.00.
    """
    roi = crop_bubble(gray, cx, cy, r, extra=1.2)
    if roi is None or roi.size == 0:
        return 0.0, 255.0

    # Resize ve chuan 60x60
    target = 60
    roi_resized = cv2.resize(roi, (target, target), interpolation=cv2.INTER_AREA)

    cx_r = target // 2
    cy_r = target // 2
    rad  = int(target * 0.22)
    mask = np.zeros((target, target), dtype=np.uint8)
    cv2.circle(mask, (cx_r, cy_r), rad, 255, -1)

    # --- Chi so 1: fill_ratio theo nguong co dinh
    _, thresh = cv2.threshold(roi_resized, 140, 255, cv2.THRESH_BINARY_INV)
    masked = cv2.bitwise_and(thresh, thresh, mask=mask)
    filled = cv2.countNonZero(masked)
    total  = cv2.countNonZero(mask)
    fill_ratio = filled / total if total > 0 else 0.0

    # --- Chi so 2: darkness = 1 - mean / 255
    mean_gray = float(cv2.mean(roi_resized, mask=mask)[0])
    darkness = max(0.0, min(1.0, (240.0 - mean_gray) / 190.0))

    # --- Chi so 3: local adaptive (so sanh voi mean cua ROI ngoai)
    # Lay mean cua cac pixel O VANH NGOAI (vung chu yeu la giay trang)
    outer_mask = np.ones((target, target), dtype=np.uint8) * 255
    cv2.circle(outer_mask, (cx_r, cy_r), int(target * 0.38), 0, -1)
    # Chi lay vung goc (background)
    corner_mask = np.zeros((target, target), dtype=np.uint8)
    corner_mask[:8, :8] = 255
    corner_mask[:8, -8:] = 255
    corner_mask[-8:, :8] = 255
    corner_mask[-8:, -8:] = 255
    bg_mean = float(cv2.mean(roi_resized, mask=corner_mask)[0])
    # Nếu trung tâm tối hơn nền 30+ đơn vị => có tô
    local_delta = max(0.0, (bg_mean - mean_gray))
    local_adapt = max(0.0, min(1.0, local_delta / 100.0))

    # Tra ve gia tri cao nhat trong 3 chi so (robust voi anh mo)
    ratio = max(fill_ratio, darkness, local_adapt)
    return ratio, mean_gray


def bubble_score(gray, cx, cy, r):
    """Diem so don gian."""
    ratio, _ = bubble_fill_ratio(gray, cx, cy, r)
    return ratio


# ---------------------------------------------------------------------------
# HAM QUY TAC CHON DAP AN TU DIEM SO
# ---------------------------------------------------------------------------
def pick_one(scores, fill_thresh=0.45, gap_thresh=0.18, weak_thresh=0.28):
    """Chon bubble duoc to trong mot nhom bubble."""
    if not scores:
        return None, 'empty'
    max_v = max(scores)
    sorted_s = sorted(scores, reverse=True)
    gap = sorted_s[0] - sorted_s[1] if len(sorted_s) > 1 else sorted_s[0]
    idx = int(np.argmax(scores))

    if max_v >= fill_thresh and gap >= gap_thresh:
        return idx, 'filled'

    if max_v >= weak_thresh and gap >= gap_thresh / 2:
        return idx, 'weak'

    if max_v < weak_thresh:
        return None, 'empty'

    return None, 'uncertain'


def pick_binary(score_a, score_b, fill_thresh=0.40, gap_thresh=0.15, weak_thresh=0.25):
    """Chon giua 2 bubble (vd: Dung/Sai)."""
    return pick_one([score_a, score_b],
                    fill_thresh=fill_thresh,
                    gap_thresh=gap_thresh,
                    weak_thresh=weak_thresh)
