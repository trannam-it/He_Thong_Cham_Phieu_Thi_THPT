# =============================================================================
# ALIGNMENT - Phat hien fiducial markers & ap dung perspective warp
# ---------------------------------------------------------------------------
# PHIEN BAN v5 (FIX SKEW SAU WARP):
#
# CHIEN LUOC MOI - SUBPIXEL + MULTI-MARKER HOMOGRAPHY:
#   1. Tien xu ly anh nang cao (CLAHE, denoise) - giu nguyen.
#   2. Phat hien TAT CA marker doc theo canh phai/trai voi do chinh xac
#      SUBPIXEL (dung center-of-mass tu anh xam thay vi bounding box).
#   3. Fit duong thang RANSAC qua cac marker phai (canh dam nhat) de xac
#      dinh huong canh phai chinh xac, sau do tinh TR/BR la marker dau/
#      cuoi tren duong nay.
#   4. Lam tuong tu cho canh trai (neu du marker), nguoc lai PROJECT TL/BL
#      bang cach giu khoang cach trai-phai song song voi canh phai.
#   5. WARP LAN DAU bang 4 goc cai thien.
#   6. TINH CHINH HOMOGRAPHY: re-detect marker phai tren anh warp, ghep cap
#      voi TEMPLATE_RIGHT_MARKER_YS, dung cv2.findHomography RANSAC voi
#      ~40 cap diem -> sua het skew con sot lai.
#   7. (Tuy chon) ECC affine refinement voi anh mau.
#
# ----- KHAC BIET CHINH SO VOI v4 -----
#   - Subpixel center-of-mass (chinh xac < 0.5 px) thay vi (x+w/2, y+h/2)
#   - Line fitting RANSAC qua nhieu marker (chong outlier)
#   - Multi-point homography refinement DUNG TAT CA marker phai sau warp
#     -> sua skew tu 5 deg -> < 0.2 deg
#   - Tighter validation cho corner candidate (loai stray markers)
# =============================================================================

import cv2
import os
import numpy as np
from .template1 import (
    TEMPLATE_WIDTH, TEMPLATE_HEIGHT,
    TEMPLATE_FIDUCIAL_TL, TEMPLATE_FIDUCIAL_TR,
    TEMPLATE_FIDUCIAL_BR, TEMPLATE_FIDUCIAL_BL,
    TEMPLATE_RIGHT_MARKER_X, TEMPLATE_RIGHT_MARKER_YS,
    TEMPLATE_LEFT_MARKER_X, TEMPLATE_LEFT_MARKER_YS,
    FIDUCIAL_MIN_W, FIDUCIAL_MAX_W,
    FIDUCIAL_MIN_H, FIDUCIAL_MAX_H,
    FIDUCIAL_MIN_AR, FIDUCIAL_MAX_AR,
    FIDUCIAL_EDGE_MARGIN_RATIO,
)


# Lưu cache ảnh mẫu (template reference image) để không load lại nhiều lần
_REFERENCE_IMAGE_CACHE = {}


def get_reference_image(base_dir=None):
    """Đọc ảnh mẫu từ Anh_cham/Anh_mau_phieu/ để làm template matching."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ref_dir = os.path.join(base_dir, "Anh_cham", "Anh_mau_phieu")
    if not os.path.isdir(ref_dir):
        return None

    if ref_dir in _REFERENCE_IMAGE_CACHE:
        return _REFERENCE_IMAGE_CACHE[ref_dir]

    files = [f for f in os.listdir(ref_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not files:
        _REFERENCE_IMAGE_CACHE[ref_dir] = None
        return None

    ref_path = os.path.join(ref_dir, sorted(files)[0])
    try:
        data = np.fromfile(ref_path, dtype=np.uint8)
        ref_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if ref_img is None:
            _REFERENCE_IMAGE_CACHE[ref_dir] = None
            return None

        _, ref_warped_gray, _ = align_to_template(
            ref_img, base_dir=None, use_reference=False
        )
        _REFERENCE_IMAGE_CACHE[ref_dir] = ref_warped_gray
        print(f"  [REF] Đã load ảnh mẫu: {ref_path}")
        return ref_warped_gray
    except Exception as e:
        print(f"  [REF] Lỗi đọc ảnh mẫu: {e}")
        _REFERENCE_IMAGE_CACHE[ref_dir] = None
        return None


# ---------------------------------------------------------------------------
# BƯỚC 0: TIỀN XỬ LÝ NÂNG CAO cho ảnh mờ, tối không đều
# ---------------------------------------------------------------------------
def enhance_image_quality(img_color):
    """Nâng cao chất lượng ảnh trước khi xử lý."""
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=5,
                                         templateWindowSize=7,
                                         searchWindowSize=21)

    enhanced_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr, denoised


# ---------------------------------------------------------------------------
# UTILITIES - sap xep diem va kiem tra hop le
# ---------------------------------------------------------------------------
def _order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự TL, TR, BR, BL."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _quad_area(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    x = pts[:, 0]; y = pts[:, 1]
    return 0.5 * abs(x[0]*(y[1]-y[3]) + x[1]*(y[2]-y[0])
                     + x[2]*(y[3]-y[1]) + x[3]*(y[0]-y[2]))


def _is_reasonable_quad(pts, img_w, img_h, min_area_ratio=0.35,
                        min_aspect_ratio=0.5, max_aspect_ratio=1.2):
    img_area = float(img_w * img_h)
    if img_area <= 0:
        return False

    quad_area = _quad_area(pts)
    if quad_area / img_area < min_area_ratio:
        return False

    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)

    avg_w = (w_top + w_bot) / 2.0
    avg_h = (h_left + h_right) / 2.0
    if avg_w <= 1 or avg_h <= 1:
        return False

    aspect = avg_w / avg_h
    if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
        return False

    pts_int = pts.astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(pts_int):
        return False

    return True


# ---------------------------------------------------------------------------
# BINARIZATION - tao anh nhi phan robust de bat marker
# ---------------------------------------------------------------------------
def _multi_threshold_binary(gray):
    """Tao anh nhi phan voi nhieu nguong de bat marker trong moi dieu kien."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw_otsu = cv2.threshold(blur, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw_adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    _, bw_fixed1 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
    _, bw_fixed2 = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.bitwise_or(cv2.bitwise_or(bw_otsu, bw_adaptive),
                         cv2.bitwise_or(bw_fixed1, bw_fixed2))
    return bw


def _subpixel_center(gray, contour, x, y, w, h):
    """Tinh tam marker voi do chinh xac SUBPIXEL bang center-of-mass tren
    vung anh xam (1 - normalized intensity).

    Tra ve (cx_subpixel, cy_subpixel).
    """
    H, W = gray.shape
    pad = 2
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return float(x + w / 2.0), float(y + h / 2.0)
    # Marker la mau DEN -> intensity thap. Lay 255 - roi de moments lay max o tam.
    weight = 255.0 - roi
    weight[weight < 30] = 0  # bo nen sang
    s = weight.sum()
    if s < 1e-3:
        return float(x + w / 2.0), float(y + h / 2.0)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cx = float((weight * xs).sum() / s)
    cy = float((weight * ys).sum() / s)
    return cx, cy


def _detect_edge_markers_raw(gray):
    """Phat hien tat ca marker tiem nang gan canh trai/phai cua anh.

    DUNG SUBPIXEL CENTER-OF-MASS de tam chinh xac.
    Tra ve (left_markers, right_markers) - moi marker la dict
    {'cx', 'cy', 'w', 'h', 'x_anchor'}.
    """
    H, W = gray.shape
    bw = _multi_threshold_binary(gray)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale = max(1.0, W / 1100.0)
    margin_x = int(W * 0.18)

    left_m, right_m = [], []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h <= 0 or w <= 0:
            continue
        if not (8 * scale <= w <= 80 * scale):
            continue
        if not (3 * scale <= h <= 30 * scale):
            continue
        ar = w / float(h)
        if not (1.3 <= ar <= 12.0):
            continue
        # Subpixel center-of-mass
        cx, cy = _subpixel_center(gray, c, x, y, w, h)
        if x < margin_x:
            left_m.append((cx, cy, w, h, x))
        elif x + w > W - margin_x:
            right_m.append((cx, cy, w, h, x + w))

    return left_m, right_m


# ---------------------------------------------------------------------------
# HAM HO TRO: FIT DUONG THANG RANSAC QUA NHIEU DIEM
# ---------------------------------------------------------------------------
def _ransac_line_fit(points, n_iter=200, dist_thr=3.0):
    """Fit duong thang qua mot tap diem (cx, cy) bang RANSAC.

    Tra ve (a, b, c) sao cho a*x + b*y + c = 0, va danh sach inliers.
    Neu khong du diem, tra ve None.
    """
    if len(points) < 2:
        return None, []
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    n = len(pts)
    if n == 2:
        # 2 diem -> 1 duong duy nhat
        p1, p2 = pts[0], pts[1]
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        a, b = dy, -dx
        norm = (a * a + b * b) ** 0.5
        if norm < 1e-6:
            return None, []
        a /= norm; b /= norm
        c = -(a * p1[0] + b * p1[1])
        return (float(a), float(b), float(c)), list(range(n))

    rng = np.random.default_rng(42)
    best_inliers = []
    best_line = None
    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        p1, p2 = pts[i], pts[j]
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        norm_d = (dx * dx + dy * dy) ** 0.5
        if norm_d < 1e-3:
            continue
        a, b = dy / norm_d, -dx / norm_d
        c = -(a * p1[0] + b * p1[1])
        # Distance from each point to line
        d = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
        inliers = np.where(d < dist_thr)[0].tolist()
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (float(a), float(b), float(c))
            if len(inliers) >= 0.95 * n:
                break

    if best_line is None or len(best_inliers) < 2:
        return None, []

    # Refit duong qua tat ca inliers bang least-squares (cv2.fitLine)
    inlier_pts = pts[best_inliers]
    [vx, vy, x0, y0] = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(vx); vy = float(vy); x0 = float(x0); y0 = float(y0)
    norm_d = (vx * vx + vy * vy) ** 0.5
    a = vy / norm_d
    b = -vx / norm_d
    c = -(a * x0 + b * y0)
    return (a, b, c), best_inliers


def _project_y_to_line(line_abc, y):
    """Cho duong a*x + b*y + c = 0, tinh x ung voi y cho truoc."""
    a, b, c = line_abc
    if abs(a) < 1e-6:
        return None
    return -(b * y + c) / a


def _line_angle_deg(line_abc):
    """Goc cua duong so voi truc Y (deg). 0 = doc thang dung, 90 = ngang."""
    a, b, _ = line_abc
    # vector chi phuong: (-b, a). Goc voi truc Y: atan2(-b, a) - 90...
    # Don gian: do nghieng so voi truc Y la atan(b/a) (vi a*x + b*y = 0
    # -> x = -b/a * y -> slope dx/dy = -b/a)
    if abs(a) < 1e-6:
        return 90.0
    slope = -b / a  # dx/dy
    return float(np.degrees(np.arctan(slope)))


# ---------------------------------------------------------------------------
# CHIEN LUOC A: TIM 4 GOC PHIEU BANG MARKER - V8 (RANSAC LINE FIT)
# ---------------------------------------------------------------------------
def _filter_edge_markers_by_line(markers, is_left, img_W):
    """Loc danh sach marker theo duong thang RANSAC.

    Tra ve (filtered_markers, line_abc, edge_x_at_mid).
    """
    if len(markers) < 2:
        return markers, None, (markers[0][4] if markers else 0.0)

    pts = np.array([[m[0], m[1]] for m in markers], dtype=np.float32)
    line, inlier_idx = _ransac_line_fit(pts.tolist(), n_iter=300, dist_thr=4.0)
    if line is None:
        return markers, None, float(np.mean([m[4] for m in markers]))

    filtered = [markers[i] for i in inlier_idx]
    if not filtered:
        return markers, line, float(np.mean([m[4] for m in markers]))

    # X tai diem giua anh (theo y)
    H_est = max(m[1] for m in filtered) - min(m[1] for m in filtered)
    y_mid = (max(m[1] for m in filtered) + min(m[1] for m in filtered)) / 2.0
    x_mid = _project_y_to_line(line, y_mid)
    if x_mid is None:
        x_mid = float(np.mean([m[0] for m in filtered]))

    return filtered, line, float(x_mid)


def _filter_left_markers_strict(left_m, ref_x_top, ref_x_bot, ref_y_top, ref_y_bot,
                                  expected_dx_top, expected_dx_bot, tol_x=80):
    """Loc marker trai NGHIEM NGAT: chi giu marker o vi tri x du kien
    (ref_x - expected_dx) trong pham vi tol_x.

    Tra ve list marker da loc, va edge_x trung binh.
    """
    if not left_m:
        return [], None
    expected_x_top = ref_x_top - expected_dx_top
    expected_x_bot = ref_x_bot - expected_dx_bot
    valid = []
    for m in left_m:
        cx, cy = m[0], m[1]
        # Noi suy x du kien tai y = cy
        if abs(ref_y_bot - ref_y_top) < 1e-3:
            t = 0
        else:
            t = (cy - ref_y_top) / (ref_y_bot - ref_y_top)
        exp_x = expected_x_top + t * (expected_x_bot - expected_x_top)
        if abs(cx - exp_x) <= tol_x:
            valid.append(m)
    if not valid:
        return [], None
    edge_x = float(np.mean([m[4] for m in valid]))
    return valid, edge_x


def _estimate_corner_y_from_markers(detected_ys, template_ys):
    """Tu danh sach y cua marker da phat hien tren canh phai/trai (sau khi
    da loc bang RANSAC), uoc luong y CUA HAI MARKER GOC (tuong ung voi
    template_ys[0] = TR/TL va template_ys[-1] = BR/BL).

    Cach lam: xet moi cap (i, j) trong template_ys, gia su detected_ys[0]
    map den template_ys[i] va detected_ys[-1] map den template_ys[j]. Tinh
    scale + offset. Sau do tinh tong khoang cach toi gan nhat tu moi
    detected_y den template_ys map. Cap (i, j) cho sai so nho nhat la dung.

    Tra ve (y_corner_top_in_image, y_corner_bot_in_image, scale, offset).
    Neu khong tim duoc, tra ve (None, None, None, None).
    """
    if len(detected_ys) < 4 or len(template_ys) < 4:
        return None, None, None, None

    detected_ys = sorted(detected_ys)
    template_ys = sorted(template_ys)
    n = len(template_ys)

    d_top = detected_ys[0]
    d_bot = detected_ys[-1]
    d_arr = np.asarray(detected_ys, dtype=np.float32)
    t_arr = np.asarray(template_ys, dtype=np.float32)

    best_err = float('inf')
    best_i = 0
    best_j = n - 1

    # Limit hypothesis: detected_ys[0] map vao template_ys[0..min(8,n-3)]
    # (vi co the bi mat may marker dau). Tuong tu cho bot.
    max_skip_top = min(8, n // 2)
    max_skip_bot = min(8, n // 2)

    for i in range(0, max_skip_top + 1):
        for j in range(n - 1, n - 1 - max_skip_bot - 1, -1):
            if j <= i:
                continue
            t_top = template_ys[i]
            t_bot = template_ys[j]
            if t_bot - t_top < 1e-3:
                continue
            scale = (d_bot - d_top) / (t_bot - t_top)
            if scale <= 0:
                continue
            offset = d_top - scale * t_top
            # Map all template_ys -> image space
            mapped = scale * t_arr + offset
            # For each detected y, find min distance to mapped
            err = 0.0
            for dy in d_arr:
                err += float(np.min(np.abs(mapped - dy)))
            if err < best_err:
                best_err = err
                best_i = i
                best_j = j

    # Best: reconstruct with chosen i, j
    i, j = best_i, best_j
    t_top = template_ys[i]
    t_bot = template_ys[j]
    scale = (d_bot - d_top) / (t_bot - t_top)
    offset = d_top - scale * t_top

    # Reconstruct y of FIRST template (template_ys[0]) and LAST (template_ys[-1])
    # in image space
    y_corner_top = scale * template_ys[0] + offset
    y_corner_bot = scale * template_ys[-1] + offset
    return float(y_corner_top), float(y_corner_bot), float(scale), float(offset)


def find_corners_by_markers(gray):
    """V8.2 FINAL: Tim 4 goc phieu bang fiducial markers.

    CHIEN LUOC:
      1. Tin tuong CANH PHAI (40 marker) lam REFERENCE TUYET DOI.
      2. Fit duong thang RANSAC cho canh phai.
      3. UOC LUONG y CUA TR/BR CORNER bang cach matching marker da phat
         hien voi TEMPLATE_RIGHT_MARKER_YS (chia te khi top/bottom corner
         marker bi mat phat hien).
      4. Tu canh phai, du doan vi tri X cua canh trai.
      5. Loc marker trai. Neu co marker hop le -> dung. Khong -> PROJECT.

    Tra ve (corners_4, success). Corners theo thu tu TL, TR, BR, BL.
    """
    H, W = gray.shape

    left_m, right_m = _detect_edge_markers_raw(gray)
    if len(right_m) < 2 and len(left_m) < 2:
        return None, False

    # ---- 1. Phan tich canh phai (uu tien tuyet doi) ----
    right_f = []
    right_line = None
    right_x = 0.0
    if len(right_m) >= 2:
        right_f, right_line, right_x = _filter_edge_markers_by_line(right_m, False, W)

    use_right_as_ref = (
        len(right_f) >= 3
        and right_line is not None
        and (max(m[1] for m in right_f) - min(m[1] for m in right_f)) >= 0.4 * H
    )

    if not use_right_as_ref:
        if len(left_m) >= 2:
            left_f, left_line, left_x = _filter_edge_markers_by_line(left_m, True, W)
            if (len(left_f) >= 3 and left_line is not None
                and (max(m[1] for m in left_f) - min(m[1] for m in left_f)) >= 0.4 * H):
                return _find_corners_with_left_ref(left_f, left_line, left_x,
                                                    right_m, H, W)
        return None, False

    # ---- 2. Use RIGHT side as reference ----
    right_f = sorted(right_f, key=lambda m: m[1])

    # 2.a UOC LUONG y CUA HAI MARKER GOC (TR, BR) bang pattern matching
    detected_ys = [m[1] for m in right_f]
    y_tr_est, y_br_est, scale, offset = _estimate_corner_y_from_markers(
        detected_ys, TEMPLATE_RIGHT_MARKER_YS
    )

    if y_tr_est is None:
        # Fallback: dung topmost & bottommost detected marker
        ref_top = right_f[0]
        ref_bot = right_f[-1]
        tr = (float(ref_top[0]), float(ref_top[1]))
        br = (float(ref_bot[0]), float(ref_bot[1]))
    else:
        # Use estimated TR, BR. X la noi suy tren duong RANSAC.
        if right_line is not None:
            x_tr = _project_y_to_line(right_line, y_tr_est)
            x_br = _project_y_to_line(right_line, y_br_est)
            if x_tr is None: x_tr = right_f[0][0]
            if x_br is None: x_br = right_f[-1][0]
        else:
            x_tr = right_f[0][0]
            x_br = right_f[-1][0]
        tr = (float(x_tr), float(y_tr_est))
        br = (float(x_br), float(y_br_est))

    # ---- 3. Du doan vi tri canh trai dua tren canh phai ----
    # Trong template: TL_x=54, TR_x=1036, do rong = 982
    # Ratio: vi tri canh trai = canh phai - 982 / (TEMPLATE_WIDTH * scale_W)
    # Scale W cua anh hien tai = right_x_at_y / 1036_template * TEMPLATE_WIDTH
    # don gian: gia su anh chua co distorsion lon, du doan x_left = x_right - K
    # voi K duoc uoc luong tu right_x.
    # Tu phieu chuan: marker phai cach mep phai khoang (1100-1036)/1100 = 5.8% chieu rong
    # -> chieu rong ROI = right_x / (1 - 0.058) = right_x / 0.942
    # -> chieu rong tu canh trai den canh phai = ROI * (1036-54)/1100 = ROI * 0.893
    # -> canh trai = right_x - ROI * 0.893
    if right_x > 0:
        roi_w = right_x / (1 - (TEMPLATE_WIDTH - TEMPLATE_FIDUCIAL_TR[0]) / TEMPLATE_WIDTH)
        expected_dx = roi_w * (TEMPLATE_FIDUCIAL_TR[0] - TEMPLATE_FIDUCIAL_TL[0]) / TEMPLATE_WIDTH
    else:
        expected_dx = W * (TEMPLATE_FIDUCIAL_TR[0] - TEMPLATE_FIDUCIAL_TL[0]) / TEMPLATE_WIDTH

    # x du kien tai y top va y bot cua right (theo huong canh phai - giu khoang cach hang)
    expected_x_top = tr[0] - expected_dx
    expected_x_bot = br[0] - expected_dx

    # ---- 4. Loc marker trai theo vi tri du kien ----
    tol_x = max(40, int(0.06 * W))  # 6% chieu rong anh
    left_valid, left_x_avg = _filter_left_markers_strict(
        left_m, tr[0], br[0], tr[1], br[1],
        expected_dx, expected_dx, tol_x
    )

    # ---- 5. Tim TL/BL ----
    y_tol = max(40, int(0.05 * H))
    tl_cand = None
    bl_cand = None
    for m in left_valid:
        if abs(m[1] - tr[1]) <= y_tol:
            if tl_cand is None or abs(m[1] - tr[1]) < abs(tl_cand[1] - tr[1]):
                tl_cand = m
        if abs(m[1] - br[1]) <= y_tol:
            if bl_cand is None or abs(m[1] - br[1]) < abs(bl_cand[1] - br[1]):
                bl_cand = m

    # PROJECT neu khong tim duoc
    if tl_cand is not None:
        tl = (float(tl_cand[0]), float(tl_cand[1]))
    else:
        tl = (float(expected_x_top), float(tr[1]))

    if bl_cand is not None:
        bl = (float(bl_cand[0]), float(bl_cand[1]))
    else:
        bl = (float(expected_x_bot), float(br[1]))

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    if not _is_reasonable_quad(corners, W, H,
                                min_area_ratio=0.30,
                                min_aspect_ratio=0.4,
                                max_aspect_ratio=1.4):
        return None, False

    return corners, True


def _find_corners_with_left_ref(left_f, left_line, left_x, right_m, H, W):
    """Fallback: dung canh trai lam reference khi canh phai bi cat."""
    left_f = sorted(left_f, key=lambda m: m[1])
    ref_top = left_f[0]
    ref_bot = left_f[-1]
    tl = (float(ref_top[0]), float(ref_top[1]))
    bl = (float(ref_bot[0]), float(ref_bot[1]))

    if left_x > 0:
        roi_w = (W - left_x) / (1 - (TEMPLATE_WIDTH - TEMPLATE_FIDUCIAL_TR[0]) / TEMPLATE_WIDTH)
        expected_dx = roi_w * (TEMPLATE_FIDUCIAL_TR[0] - TEMPLATE_FIDUCIAL_TL[0]) / TEMPLATE_WIDTH
    else:
        expected_dx = W * (TEMPLATE_FIDUCIAL_TR[0] - TEMPLATE_FIDUCIAL_TL[0]) / TEMPLATE_WIDTH

    expected_x_top = tl[0] + expected_dx
    expected_x_bot = bl[0] + expected_dx

    # Loc right markers theo vi tri du kien
    tol_x = max(40, int(0.06 * W))
    right_valid = []
    for m in right_m:
        cy = m[1]
        if abs(bl[1] - tl[1]) < 1e-3:
            t = 0
        else:
            t = (cy - tl[1]) / (bl[1] - tl[1])
        exp_x = expected_x_top + t * (expected_x_bot - expected_x_top)
        if abs(m[0] - exp_x) <= tol_x:
            right_valid.append(m)

    y_tol = max(40, int(0.05 * H))
    tr_cand = None
    br_cand = None
    for m in right_valid:
        if abs(m[1] - tl[1]) <= y_tol:
            if tr_cand is None or abs(m[1] - tl[1]) < abs(tr_cand[1] - tl[1]):
                tr_cand = m
        if abs(m[1] - bl[1]) <= y_tol:
            if br_cand is None or abs(m[1] - bl[1]) < abs(br_cand[1] - bl[1]):
                br_cand = m

    if tr_cand is not None:
        tr = (float(tr_cand[0]), float(tr_cand[1]))
    else:
        tr = (float(expected_x_top), float(tl[1]))
    if br_cand is not None:
        br = (float(br_cand[0]), float(br_cand[1]))
    else:
        br = (float(expected_x_bot), float(bl[1]))

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    if not _is_reasonable_quad(corners, W, H,
                                min_area_ratio=0.30,
                                min_aspect_ratio=0.4,
                                max_aspect_ratio=1.4):
        return None, False
    return corners, True


# ---------------------------------------------------------------------------
# CHIEN LUOC B: TIM 4 GOC TO GIAY BANG CONTOUR LON NHAT
# ---------------------------------------------------------------------------
def find_paper_corners_by_contour(img_color, min_area_ratio=0.30):
    """Phat hien 4 goc cua to giay tren nen bang Canny + contour."""
    H, W = img_color.shape[:2]
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) \
        if len(img_color.shape) == 3 else img_color

    candidates = []
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    for low_thr, high_thr in [(30, 90), (50, 150), (20, 60)]:
        edges = cv2.Canny(blurred, low_thr, high_thr)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_d = cv2.dilate(edges, kernel, iterations=2)

        cnts, _ = cv2.findContours(
            edges_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            continue

        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        for cnt in cnts_sorted[:5]:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < W * H * min_area_ratio:
                continue
            for eps_f in [0.01, 0.02, 0.03, 0.05, 0.07]:
                eps = eps_f * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)
                if len(approx) == 4:
                    corners = _order_points(approx)
                    if _is_reasonable_quad(corners, W, H,
                                            min_area_ratio=min_area_ratio):
                        candidates.append((cnt_area, corners))
                    break
            hull = cv2.convexHull(cnt)
            for eps_f in [0.02, 0.03, 0.05, 0.07, 0.1]:
                eps = eps_f * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, eps, True)
                if len(approx) == 4:
                    corners = _order_points(approx)
                    if _is_reasonable_quad(corners, W, H,
                                            min_area_ratio=min_area_ratio):
                        candidates.append((cnt_area, corners))
                    break

    if not candidates:
        return None, False

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], True


# ---------------------------------------------------------------------------
# CHIEN LUOC C: TIM KHUNG DEN NGOAI CUA PHIEU
# ---------------------------------------------------------------------------
def find_paper_corners_by_outer_box(img_color):
    """Tim 4 goc bang quet ROW/COLUMN cua anh."""
    H, W = img_color.shape[:2]
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) \
        if len(img_color.shape) == 3 else img_color

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    row_sum = (bw > 0).sum(axis=1).astype(np.float32) / W
    col_sum = (bw > 0).sum(axis=0).astype(np.float32) / H

    thr = 0.03
    rows_with_content = np.where(row_sum > thr)[0]
    cols_with_content = np.where(col_sum > thr)[0]

    if len(rows_with_content) < 10 or len(cols_with_content) < 10:
        return None, False

    y_top = max(0, int(rows_with_content[0]) - 2)
    y_bot = min(H - 1, int(rows_with_content[-1]) + 2)
    x_left = max(0, int(cols_with_content[0]) - 2)
    x_right = min(W - 1, int(cols_with_content[-1]) + 2)

    if (y_bot - y_top) < H * 0.5 or (x_right - x_left) < W * 0.5:
        return None, False

    corners = np.array([
        [x_left, y_top], [x_right, y_top],
        [x_right, y_bot], [x_left, y_bot],
    ], dtype=np.float32)

    if not _is_reasonable_quad(corners, W, H, min_area_ratio=0.25):
        return None, False

    return corners, True


# ---------------------------------------------------------------------------
# TONG HOP: TIM 4 GOC PHIEU (KET HOP NHIEU CHIEN LUOC)
# ---------------------------------------------------------------------------
def find_sheet_corners(img_color):
    """Tim 4 goc cua phieu/to giay theo thu tu uu tien:

      1. Fiducial markers (chinh xac nhat - dung khi anh ro net).
      2. Contour to giay (anh chup co nen).
      3. Outer bounding box (anh scan sat to giay).
      4. Fallback: 4 goc anh.
    """
    H, W = img_color.shape[:2]
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) \
        if len(img_color.shape) == 3 else img_color

    # Chien luoc 1: Fiducial markers
    corners, ok = find_corners_by_markers(gray)
    if ok:
        return corners, "markers"

    # Chien luoc 2: Contour to giay
    corners, ok = find_paper_corners_by_contour(img_color,
                                                  min_area_ratio=0.30)
    if ok:
        return corners, "paper_contour"

    # Chien luoc 3: Bounding box noi dung
    corners, ok = find_paper_corners_by_outer_box(img_color)
    if ok:
        return corners, "outer_box"

    # Fallback
    corners = np.array([
        [0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]
    ], dtype=np.float32)
    return corners, "fallback_image_corners"


# ---------------------------------------------------------------------------
# WARP MOT LAN DUY NHAT VE TEMPLATE 1100x1540
# ---------------------------------------------------------------------------
def warp_corners_to_template(img_color, corners):
    """Warp 4 goc -> 4 goc template (TL, TR, BR, BL)."""
    src_pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    dst_pts = np.array([
        TEMPLATE_FIDUCIAL_TL,
        TEMPLATE_FIDUCIAL_TR,
        TEMPLATE_FIDUCIAL_BR,
        TEMPLATE_FIDUCIAL_BL,
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_color = cv2.warpPerspective(
        img_color, M, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255))
    if len(warped_color.shape) == 3:
        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    else:
        warped_gray = warped_color
    return warped_color, warped_gray, M


def _compute_skew_angle(corners):
    """Tinh goc nghieng cua tu giac (deg)."""
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]
    angles = []
    angles.append(abs(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))))
    angles.append(abs(np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))))
    a_left = abs(np.degrees(np.arctan2(bl[0] - tl[0], bl[1] - tl[1])))
    a_right = abs(np.degrees(np.arctan2(br[0] - tr[0], br[1] - tr[1])))
    angles.append(a_left); angles.append(a_right)
    return float(max(angles))


# ---------------------------------------------------------------------------
# REFINE BANG MULTI-MARKER HOMOGRAPHY (CHINH XAC NHAT)
# ---------------------------------------------------------------------------
def _detect_warped_markers(warped_gray):
    """Phat hien marker doc canh phai/trai cua anh DA WARP ve template
    1100x1540. Tra ve list-of-(cx, cy) cho moi ben.
    """
    H, W = warped_gray.shape
    bw = _multi_threshold_binary(warped_gray)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    margin_x = int(W * 0.08)  # Chi can sat mep
    right_pts = []
    left_pts = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h <= 0 or w <= 0:
            continue
        if not (8 <= w <= 60):
            continue
        if not (3 <= h <= 25):
            continue
        ar = w / float(h)
        if not (1.3 <= ar <= 12.0):
            continue
        cx, cy = _subpixel_center(warped_gray, c, x, y, w, h)
        if x < margin_x:
            left_pts.append((cx, cy))
        elif x + w > W - margin_x:
            right_pts.append((cx, cy))

    return left_pts, right_pts


def _match_markers_by_y(detected_pts, template_ys, max_dy=15.0):
    """Ghep cap detected markers (cx,cy) voi template_ys dua tren cy gan nhat.

    Tra ve (src_pts, dst_pts) cho cv2.findHomography.
    src_pts: [(cx, cy), ...] tren anh warp
    dst_pts: [(template_x, template_y), ...]
    """
    src_pts = []
    dst_pts = []
    used_template = set()
    # Sort detected by y
    detected_sorted = sorted(detected_pts, key=lambda p: p[1])
    for cx, cy in detected_sorted:
        # Find closest template_y not yet used
        best_idx = -1
        best_dy = max_dy
        for i, ty in enumerate(template_ys):
            if i in used_template:
                continue
            dy = abs(cy - ty)
            if dy < best_dy:
                best_dy = dy
                best_idx = i
        if best_idx >= 0:
            used_template.add(best_idx)
            src_pts.append((cx, cy))
            dst_pts.append(None)  # placeholder, will set later
            dst_pts[-1] = (None, template_ys[best_idx])
    return src_pts, dst_pts, list(used_template)


def refine_by_multi_marker_homography(warped_color, warped_gray):
    """TINH CHINH HOMOGRAPHY dung TAT CA marker phat hien duoc tren anh
    da warp, ghep voi marker chuan tren template.

    Day la BUOC QUAN TRONG NHAT de sua skew con sot lai sau warp 4 goc.
    Tra ve (refined_color, refined_gray, note).
    """
    H, W = warped_gray.shape
    if (H, W) != (TEMPLATE_HEIGHT, TEMPLATE_WIDTH):
        return warped_color, warped_gray, ""

    left_pts, right_pts = _detect_warped_markers(warped_gray)

    src_pts = []
    dst_pts = []

    # Match right markers vs TEMPLATE_RIGHT_MARKER_YS
    if len(right_pts) >= 4:
        for cx, cy in right_pts:
            # Find closest template y
            best_dy = 12.0
            best_y = None
            for ty in TEMPLATE_RIGHT_MARKER_YS:
                dy = abs(cy - ty)
                if dy < best_dy:
                    best_dy = dy
                    best_y = ty
            if best_y is not None:
                src_pts.append([cx, cy])
                dst_pts.append([float(TEMPLATE_RIGHT_MARKER_X), float(best_y)])

    # Match left markers (chi 2 marker goc, du them de stabilize)
    if len(left_pts) >= 1:
        for cx, cy in left_pts:
            best_dy = 12.0
            best_y = None
            for ty in TEMPLATE_LEFT_MARKER_YS:
                dy = abs(cy - ty)
                if dy < best_dy:
                    best_dy = dy
                    best_y = ty
            if best_y is not None:
                src_pts.append([cx, cy])
                dst_pts.append([float(TEMPLATE_LEFT_MARKER_X), float(best_y)])

    n_matched = len(src_pts)
    if n_matched < 6:
        # Khong du diem -> bo qua refine
        return warped_color, warped_gray, ""

    src_arr = np.array(src_pts, dtype=np.float32)
    dst_arr = np.array(dst_pts, dtype=np.float32)

    # Tinh do dich chuyen TRUNG BINH, neu qua nho thi bo qua de tranh nhieu
    diffs = dst_arr - src_arr
    max_diff = float(np.linalg.norm(diffs, axis=1).max())
    mean_diff = float(np.linalg.norm(diffs, axis=1).mean())

    if max_diff < 1.0:
        # Da rat sat template, khong can refine
        return warped_color, warped_gray, ""

    # Fit homography RANSAC (rat robust voi outlier)
    H_mat, inliers = cv2.findHomography(
        src_arr, dst_arr, cv2.RANSAC, ransacReprojThreshold=2.0,
        maxIters=2000, confidence=0.999
    )
    if H_mat is None:
        return warped_color, warped_gray, ""

    # Validate: kiem tra homography khong qua mat tinh chat (det != 0, scale ~ 1)
    # Lay 4 goc template, ap dung H_mat^-1 (de map tu warped -> template),
    # roi kiem tra ket qua co gan nhu giu nguyen kich thuoc.
    try:
        # Test: ap dung H_mat len trung tam anh, kiem tra do dich
        center_src = np.array([[[W / 2.0, H / 2.0]]], dtype=np.float32)
        center_dst = cv2.perspectiveTransform(center_src, H_mat)[0][0]
        center_shift = float(np.linalg.norm(center_dst - center_src[0][0]))
        if center_shift > 50:
            return warped_color, warped_gray, ""

        # Det check
        det = float(H_mat[0, 0] * H_mat[1, 1] - H_mat[0, 1] * H_mat[1, 0])
        if det <= 0.5 or det >= 2.0:
            return warped_color, warped_gray, ""
    except Exception:
        return warped_color, warped_gray, ""

    refined_color = cv2.warpPerspective(
        warped_color, H_mat, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    refined_gray = cv2.cvtColor(refined_color, cv2.COLOR_BGR2GRAY) \
        if len(refined_color.shape) == 3 else refined_color

    n_inliers = int(inliers.sum()) if inliers is not None else n_matched
    print(f"  [HOMOG] Multi-marker refine: matched={n_matched} "
          f"inliers={n_inliers} max_diff={max_diff:.1f}px "
          f"mean_diff={mean_diff:.2f}px")
    return refined_color, refined_gray, f"+homog({n_inliers}pts)"


# ---------------------------------------------------------------------------
# REFINE BANG ECC ALIGNMENT VOI ANH MAU
# ---------------------------------------------------------------------------
def refine_warp_by_reference(warped_gray, warped_color, base_dir=None):
    """Tinh chinh warp bang ECC alignment voi anh mau."""
    ref_gray = get_reference_image(base_dir)
    if ref_gray is None:
        return warped_color, warped_gray, ""

    try:
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        scale = 0.5
        small_ref = cv2.resize(ref_gray, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
        small_warp = cv2.resize(warped_gray, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

        small_ref_b = cv2.GaussianBlur(small_ref, (5, 5), 0)
        small_warp_b = cv2.GaussianBlur(small_warp, (5, 5), 0)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    50, 1e-5)
        cc, warp_matrix = cv2.findTransformECC(
            small_ref_b, small_warp_b,
            warp_matrix, warp_mode, criteria, None, 5
        )

        warp_matrix[0, 2] /= scale
        warp_matrix[1, 2] /= scale

        dx = abs(warp_matrix[0, 2])
        dy = abs(warp_matrix[1, 2])
        # Sau khi co homography refine, chi can ECC sua dich nho
        if dx > 25 or dy > 25:
            return warped_color, warped_gray, ""
        # Bo qua neu dich chuyen qua nho (khong dang refine)
        if dx < 0.5 and dy < 0.5:
            return warped_color, warped_gray, ""

        H, W = warped_gray.shape
        refined_color = cv2.warpAffine(
            warped_color, warp_matrix, (W, H),
            flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        refined_gray = cv2.warpAffine(
            warped_gray, warp_matrix, (W, H),
            flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        print(f"  [ECC] Refined warp cc={cc:.4f} dx={dx:.1f} dy={dy:.1f}")
        return refined_color, refined_gray, f"+ECC(cc={cc:.3f})"
    except cv2.error as e:
        print(f"  [ECC] ECC failed: {str(e)[:80]}")
        return warped_color, warped_gray, ""


# ---------------------------------------------------------------------------
# PIPELINE CHINH (DA SUA LOI - CHI WARP MOT LAN, REFINE BANG MULTI-MARKER)
# ---------------------------------------------------------------------------
def align_to_template(img_color, base_dir=None, use_reference=True):
    """Pipeline chinh: enhance -> tim 4 goc -> warp 1 lan ve 1100x1540
    -> tinh chinh bang multi-marker homography -> ECC.

    LOGIC v5:
      1. Nang cao chat luong anh.
      2. Tim 4 goc phieu bang chien luoc tot nhat (markers (RANSAC line)
         > contour > outer).
      3. Warp 1 LAN DUY NHAT ve template 1100x1540.
      4. TINH CHINH HOMOGRAPHY bang ~40 marker doc canh phai (RANSAC).
         Day la BUOC QUAN TRONG NHAT de sua skew con sot lai.
      5. (Tuy chon) Refine bang ECC voi anh mau.
    """
    H, W = img_color.shape[:2]

    enhanced_color, enhanced_gray = enhance_image_quality(img_color)

    # Buoc 1: Tim 4 goc phieu
    corners, method = find_sheet_corners(enhanced_color)
    skew_angle = _compute_skew_angle(corners)

    # Buoc 2: Warp 1 lan duy nhat ve template
    warped_color, warped_gray, M = warp_corners_to_template(
        img_color, corners
    )

    # Buoc 3: TINH CHINH HOMOGRAPHY DUNG NHIEU MARKER (BAT BUOC)
    warped_color, warped_gray, homog_note = refine_by_multi_marker_homography(
        warped_color, warped_gray
    )

    # Buoc 4: Tinh chinh marker fine-tune them mot lan (4 goc) neu can
    fine_corners, fine_ok = find_corners_by_markers(warped_gray)
    refine_note = ""
    if fine_ok:
        ideal = np.array([
            TEMPLATE_FIDUCIAL_TL,
            TEMPLATE_FIDUCIAL_TR,
            TEMPLATE_FIDUCIAL_BR,
            TEMPLATE_FIDUCIAL_BL,
        ], dtype=np.float32)
        diff = np.linalg.norm(fine_corners - ideal, axis=1)
        max_diff = float(diff.max())
        # Chi can sua neu da homog van con lech > 5 px va < 30 px
        if max_diff > 5 and max_diff < 30:
            M2 = cv2.getPerspectiveTransform(fine_corners, ideal)
            warped_color = cv2.warpPerspective(
                warped_color, M2, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
            refine_note = "+marker_fine"
            print(f"  [REFINE] Marker fine-tune (max diff={max_diff:.1f}px)")

    # Buoc 5: ECC refinement (chi giup khi co dich chuyen nho)
    ecc_note = ""
    if use_reference:
        warped_color, warped_gray, ecc_note = refine_warp_by_reference(
            warped_gray, warped_color, base_dir
        )

    info = {
        'src_pts': corners,
        'dst_pts': np.array([
            TEMPLATE_FIDUCIAL_TL, TEMPLATE_FIDUCIAL_TR,
            TEMPLATE_FIDUCIAL_BR, TEMPLATE_FIDUCIAL_BL,
        ], dtype=np.float32),
        'method': method + homog_note + refine_note + ecc_note,
        'M': M,
        'scale': 1.0,
        'img_scaled': img_color,
        'skew_angle': skew_angle,
        'rotated': skew_angle > 1.0,
        'enhanced': True,
        'paper_found': method != "fallback_image_corners",
    }
    return warped_color, warped_gray, info


# ---------------------------------------------------------------------------
# BACK-COMPAT: cac ham giu nguyen API cho code goi tu ngoai
# ---------------------------------------------------------------------------
def preprocess_and_align(img_color, base_dir=None, use_reference=True):
    return align_to_template(img_color, base_dir=base_dir,
                              use_reference=use_reference)


def normalize_image(img_color):
    h0, w0 = img_color.shape[:2]
    scale = TEMPLATE_WIDTH / w0
    new_w = TEMPLATE_WIDTH
    new_h = int(round(h0 * scale))
    img_scaled = cv2.resize(
        img_color, (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    return img_scaled, gray, scale


def warp_to_template(img_color, gray):
    corners, method = find_sheet_corners(img_color)
    warped_color, warped_gray, M = warp_corners_to_template(img_color, corners)
    info = {
        'src_pts': corners,
        'dst_pts': np.array([
            TEMPLATE_FIDUCIAL_TL, TEMPLATE_FIDUCIAL_TR,
            TEMPLATE_FIDUCIAL_BR, TEMPLATE_FIDUCIAL_BL,
        ], dtype=np.float32),
        'method': method,
        'M': M,
    }
    return warped_color, warped_gray, info


def find_four_corners(gray):
    H, W = gray.shape
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    corners, method = find_sheet_corners(img_color)
    return corners, method


def detect_paper_corners(img_color):
    corners, method = find_sheet_corners(img_color)
    return corners, (method != "fallback_image_corners")


def find_edge_markers(gray):
    """Backward compat: tra ve (left, right) marker tuples (x, y, w, h)."""
    left_m, right_m = _detect_edge_markers_raw(gray)
    # Convert sang format cu: (x, y, w, h)
    left_old = [(int(m[0] - m[2]/2), int(m[1] - m[3]/2), m[2], m[3])
                for m in left_m]
    right_old = [(int(m[0] - m[2]/2), int(m[1] - m[3]/2), m[2], m[3])
                 for m in right_m]
    return left_old, right_old


def is_image_rotated(corners, tolerance_deg=2.0):
    angle = _compute_skew_angle(corners)
    return angle > tolerance_deg, angle


def warp_paper_to_rectangle(img_color, paper_corners):
    src_pts = paper_corners.astype(np.float32)
    w_top = np.linalg.norm(src_pts[1] - src_pts[0])
    w_bottom = np.linalg.norm(src_pts[2] - src_pts[3])
    dst_w = int(max(w_top, w_bottom))
    h_left = np.linalg.norm(src_pts[3] - src_pts[0])
    h_right = np.linalg.norm(src_pts[2] - src_pts[1])
    dst_h = int(max(h_left, h_right))
    if dst_w < 100 or dst_h < 100:
        return None, 0, 0
    dst_pts = np.float32([
        [0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h],
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        img_color, M, (dst_w, dst_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped, dst_w, dst_h


def deskew_image(gray):
    """DEPRECATED."""
    return gray, 0.0


def _find_edge_cluster(*args, **kwargs):
    """DEPRECATED: kept for backward compatibility."""
    return [], 0.0
