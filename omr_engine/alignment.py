# =============================================================================
# ALIGNMENT - Phat hien fiducial markers & ap dung perspective warp
# ---------------------------------------------------------------------------
# PHIEN BAN DA SUA LOI HOAN TOAN (v4 - FINAL):
#
# CHIEN LUOC MOI - DON GIAN, CHINH XAC, KHONG TICH LUY LOI:
#   1. Tien xu ly anh nang cao (CLAHE, denoise) - giu nguyen.
#   2. TIM 4 GOC PHIEU bang nhieu chien luoc THEO THU TU UU TIEN:
#        a. MULTI-THRESHOLD marker detection (Otsu + Adaptive + Fixed)
#        b. Cluster-based edge detection (chong outlier)
#        c. Reference-side strategy: ben nhieu marker -> ref, ben kia
#           tim marker khop Y, neu khong thi extrapolate
#        d. Fallback: contour to giay / 4 goc anh
#   3. WARP MOT LAN DUY NHAT ve template 1100x1540 (KHONG warp 2 lan).
#   4. Refine bang ECC voi anh mau (neu can).
#
# ----- KHAC BIET CHINH SO VOI VERSION CU -----
#   - KHONG con buoc "warp_paper_to_rectangle" trung gian (gay loi tich luy).
#   - Multi-threshold detection bat duoc markers trong nhieu dieu kien anh sang
#   - Cluster-based filter chong outlier (vach den dam o mep, ky hieu khac)
#   - Reference-side: dung ben dam dac de uoc luong Y top/bot, ben thieu
#     thi extrapolate -> ngan loi cat header/footer.
# =============================================================================

import cv2
import os
import numpy as np
from .template1 import (
    TEMPLATE_WIDTH, TEMPLATE_HEIGHT,
    TEMPLATE_FIDUCIAL_TL, TEMPLATE_FIDUCIAL_TR,
    TEMPLATE_FIDUCIAL_BR, TEMPLATE_FIDUCIAL_BL,
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
# CHIEN LUOC A: TIM 4 FIDUCIAL MARKERS GOC (4 vach DEN o 4 GOC) - V7 FINAL
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


def _detect_edge_markers_raw(gray):
    """Phat hien tat ca marker tiem nang gan canh trai/phai cua anh.

    Tra ve (left_markers, right_markers) - moi marker la
    (cx, cy, w, h, x_anchor) trong do x_anchor la x_left voi marker trai
    va x_right (=x+w) voi marker phai.
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
        cx, cy = x + w / 2.0, y + h / 2.0
        if x < margin_x:
            left_m.append((cx, cy, w, h, x))
        elif x + w > W - margin_x:
            right_m.append((cx, cy, w, h, x + w))

    return left_m, right_m


def _find_edge_cluster(markers, is_left, img_width):
    """Tim cum marker SAT MEP nhat - day la cum marker thuc su tren canh phieu.

    Cach lam: lay marker gan mep nhat lam anchor, mo rong tolerance de gom
    cac marker cung cum. Loai outlier (vach den lon don le).
    """
    if not markers:
        return [], 0.0

    scale = max(1.0, img_width / 1100.0)

    if len(markers) == 1:
        return markers, markers[0][4]

    # Sap xep theo edge_x: trai -> tang dan, phai -> giam dan
    if is_left:
        markers_sorted = sorted(markers, key=lambda m: m[4])
    else:
        markers_sorted = sorted(markers, key=lambda m: -m[4])

    tol = max(25, int(30 * scale))

    # Anchor: marker GAN MEP nhat
    anchor = markers_sorted[0]
    cluster = [m for m in markers if abs(m[4] - anchor[4]) <= tol]

    # Neu cluster qua nho (1 marker), mo rong tolerance them
    if len(cluster) == 1 and len(markers) > 1:
        tol2 = max(50, int(60 * scale))
        cluster = [m for m in markers if abs(m[4] - anchor[4]) <= tol2]

    # Loai outlier dac biet: marker co edge_x cach trung binh > 2*tol
    if len(cluster) >= 3:
        edge_xs = [m[4] for m in cluster]
        median_x = float(np.median(edge_xs))
        cluster = [m for m in cluster if abs(m[4] - median_x) <= tol]

    edge_x = float(np.mean([m[4] for m in cluster])) if cluster else 0.0
    return cluster, edge_x


def find_corners_by_markers(gray):
    """V7 FINAL: Tim 4 goc phieu bang fiducial markers.

    Tra ve (corners_4, success). Corners theo thu tu TL, TR, BR, BL.
    """
    H, W = gray.shape

    left_m, right_m = _detect_edge_markers_raw(gray)
    if len(left_m) < 1 or len(right_m) < 1:
        return None, False

    # Loc cluster gan mep nhat
    left_f, left_edge_x = _find_edge_cluster(left_m, True, W)
    right_f, right_edge_x = _find_edge_cluster(right_m, False, W)
    if len(left_f) < 1 or len(right_f) < 1:
        return None, False

    left_f = sorted(left_f, key=lambda m: m[1])
    right_f = sorted(right_f, key=lambda m: m[1])

    def y_range(side):
        return side[-1][1] - side[0][1] if side else 0

    left_range = y_range(left_f)
    right_range = y_range(right_f)

    # Chon ben tham chieu (ben dam dac, y_range rong nhat)
    if right_range >= 0.7 * H and right_range >= left_range:
        ref_side, other_side = right_f, left_f
        ref_is_right = True
    elif left_range >= 0.7 * H and left_range >= right_range:
        ref_side, other_side = left_f, right_f
        ref_is_right = False
    elif right_range >= left_range:
        ref_side, other_side = right_f, left_f
        ref_is_right = True
    else:
        ref_side, other_side = left_f, right_f
        ref_is_right = False

    ref_top_y = ref_side[0][1]
    ref_bot_y = ref_side[-1][1]

    # Reference phai cover >= 50% chieu cao anh
    if (ref_bot_y - ref_top_y) < 0.5 * H:
        return None, False

    # Tim trong other_side marker khop voi ref_top_y va ref_bot_y
    other_top = min(other_side, key=lambda m: abs(m[1] - ref_top_y))
    other_bot = min(other_side, key=lambda m: abs(m[1] - ref_bot_y))

    # Validation + Extrapolate fallback
    y_tol_strict = 0.10 * H
    other_edge = left_edge_x if ref_is_right else right_edge_x

    other_top_use = other_top
    other_bot_use = other_bot

    # Neu khong khop top, EXTRAPOLATE: x = edge_x cua ben thieu, y = ref_y
    if abs(other_top[1] - ref_top_y) > y_tol_strict:
        other_top_use = (other_edge, ref_top_y, 0, 0, other_edge)
    if abs(other_bot[1] - ref_bot_y) > y_tol_strict:
        other_bot_use = (other_edge, ref_bot_y, 0, 0, other_edge)

    if ref_is_right:
        tr = (ref_side[0][0], ref_side[0][1])
        br = (ref_side[-1][0], ref_side[-1][1])
        tl = (other_top_use[0], other_top_use[1])
        bl = (other_bot_use[0], other_bot_use[1])
    else:
        tl = (ref_side[0][0], ref_side[0][1])
        bl = (ref_side[-1][0], ref_side[-1][1])
        tr = (other_top_use[0], other_top_use[1])
        br = (other_bot_use[0], other_bot_use[1])

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    # Kiem tra hop le
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
        if dx > 50 or dy > 50:
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
# PIPELINE CHINH (DA SUA LOI - CHI WARP MOT LAN)
# ---------------------------------------------------------------------------
def align_to_template(img_color, base_dir=None, use_reference=True):
    """Pipeline chinh: enhance -> tim 4 goc -> warp 1 lan ve 1100x1540.

    LOGIC MOI (KHONG TICH LUY LOI):
      1. Nang cao chat luong anh.
      2. Tim 4 goc phieu bang chien luoc tot nhat (markers > contour > outer).
      3. Warp 1 LAN DUY NHAT ve template 1100x1540.
      4. Tinh chinh fiducial markers tren anh DA WARP (nhe).
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

    # Buoc 3: Tinh chinh markers tren anh da warp
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
        if max_diff > 4 and max_diff < 80:
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

    # Buoc 4: ECC refinement
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
        'method': method + refine_note + ecc_note,
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
