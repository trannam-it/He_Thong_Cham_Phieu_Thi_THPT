# =============================================================================
# DEBUG EXPORTER - Xuất các ảnh debug trung gian để người dùng xem và chỉnh code
# ---------------------------------------------------------------------------
# Mỗi ảnh sẽ được xuất vào thư mục Anh_da_cham/<tên_ảnh>_debug/ bao gồm:
#   - 00_original.jpg       : ảnh gốc đã load
#   - 01_enhanced.jpg       : ảnh sau khi CLAHE + unsharp + denoise
#   - 02_warped.jpg         : ảnh sau warp về template 1100x1540
#   - 03_binary_global.jpg  : ảnh nhị phân toàn ảnh (Otsu)
#   - 04_binary_adaptive.jpg: ảnh nhị phân adaptive
#   - 05_contours.jpg       : ảnh vẽ đường biên tất cả contour
#   - 10_header_crop.jpg    : vùng cắt header (SBD + MDT)
#   - 11_header_bin.jpg     : nhị phân của header
#   - 12_header_contour.jpg : đường biên của header
#   - 20_phan1_crop.jpg     : vùng cắt Phần I
#   - 21_phan1_bin.jpg      : nhị phân của Phần I
#   - 22_phan1_contour.jpg  : đường biên của Phần I
#   - 30_phan2_crop.jpg     : vùng cắt Phần II
#   - 31_phan2_bin.jpg      : nhị phân của Phần II
#   - 32_phan2_contour.jpg  : đường biên của Phần II
#   - 40_phan3_crop.jpg     : vùng cắt Phần III
#   - 41_phan3_bin.jpg      : nhị phân của Phần III
#   - 42_phan3_contour.jpg  : đường biên của Phần III
#   - 99_final_debug.jpg    : ảnh debug cuối cùng (vẽ tất cả bubble)
# =============================================================================

import cv2
import os
import numpy as np

from .template1 import (
    SBD_BBOX, MDT_BBOX,
    P1_BLOCKS, P2_BLOCKS, P3_BLOCKS,
)


def _safe_imwrite(path, img):
    """Ghi ảnh an toàn, hỗ trợ path unicode."""
    try:
        ext = os.path.splitext(path)[1] or '.jpg'
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
    except Exception as e:
        print(f"  [DEBUG] Lỗi ghi {path}: {e}")
    return False


def _binarize_global(gray):
    """Nhị phân bằng Otsu (dùng cho toàn ảnh warped)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _binarize_adaptive(gray):
    """Adaptive threshold (cho ánh sáng không đều)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )
    return bw


def _draw_contours(gray, color_img=None):
    """Tìm contour trên nhị phân và vẽ lên color_img (BGR).

    NANG CAP: dung ADAPTIVE threshold thay vi Otsu vi adaptive khu nhieu
    tot hon voi anh sang khong deu -> nhan dien block va bong dap an
    chinh xac hon.
    """
    bw = _binarize_adaptive(gray)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if color_img is None:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        vis = color_img.copy()
    # Vẽ tất cả contour với màu khác nhau
    for i, c in enumerate(cnts):
        if cv2.contourArea(c) < 5:  # bỏ qua noise rất nhỏ
            continue
        color = ((i * 37) % 255, (i * 71) % 255, (i * 113) % 255)
        cv2.drawContours(vis, [c], -1, color, 1)
    return vis


def _crop_region(img, bbox, padding=10):
    """Cắt vùng theo bbox (x1,y1,x2,y2), có padding."""
    H, W = img.shape[:2]
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(W, bbox[2] + padding)
    y2 = min(H, bbox[3] + padding)
    return img[y1:y2, x1:x2]


def _union_bbox(bboxes):
    """Gộp nhiều bbox thành bbox lớn bao tất cả."""
    if not bboxes:
        return (0, 0, 0, 0)
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return (x1, y1, x2, y2)


def export_debug_images(image_name, img_original, warped_color, warped_gray,
                         final_canvas, out_dir, info=None):
    """Xuất tất cả ảnh debug cho một phiếu.

    Args:
        image_name: tên ảnh (không có phần mở rộng)
        img_original: ảnh gốc đã đọc (BGR)
        warped_color: ảnh warped (BGR)
        warped_gray : ảnh warped grayscale
        final_canvas: ảnh debug cuối cùng (đã vẽ)
        out_dir     : thư mục gốc Anh_da_cham
        info        : info dict từ alignment (để ghi metadata)

    Trả về dict { key: path } của các ảnh debug.
    """
    debug_folder = os.path.join(out_dir, f"{image_name}_debug")
    os.makedirs(debug_folder, exist_ok=True)

    paths = {}

    # ====== GLOBAL IMAGES ======
    # 00. Ảnh gốc (đã downscale nếu quá to)
    if img_original is not None:
        H, W = img_original.shape[:2]
        if W > 1400:
            scale = 1400.0 / W
            preview = cv2.resize(img_original, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)
        else:
            preview = img_original
        p = os.path.join(debug_folder, "00_original.jpg")
        _safe_imwrite(p, preview)
        paths['original'] = p

    # 02. Ảnh warped (color)
    p = os.path.join(debug_folder, "02_warped.jpg")
    _safe_imwrite(p, warped_color)
    paths['warped'] = p

    # 03. Nhị phân Otsu toàn ảnh warped
    bw_global = _binarize_global(warped_gray)
    p = os.path.join(debug_folder, "03_binary_global.jpg")
    _safe_imwrite(p, bw_global)
    paths['binary_global'] = p

    # 04. Nhị phân adaptive
    bw_adaptive = _binarize_adaptive(warped_gray)
    p = os.path.join(debug_folder, "04_binary_adaptive.jpg")
    _safe_imwrite(p, bw_adaptive)
    paths['binary_adaptive'] = p

    # 05. Contour của toàn ảnh
    contour_img = _draw_contours(warped_gray, warped_color)
    p = os.path.join(debug_folder, "05_contours.jpg")
    _safe_imwrite(p, contour_img)
    paths['contours'] = p

    # ====== PER-SECTION: cắt, nhị phân, contour ======
    sections = [
        ("10_header", "11_header", "12_header",
         _union_bbox([SBD_BBOX, MDT_BBOX])),
        ("20_phan1", "21_phan1", "22_phan1",
         _union_bbox([b['bbox'] for b in P1_BLOCKS])),
        ("30_phan2", "31_phan2", "32_phan2",
         _union_bbox([b['bbox'] for b in P2_BLOCKS])),
        ("40_phan3", "41_phan3", "42_phan3",
         _union_bbox([b['bbox'] for b in P3_BLOCKS])),
    ]
    section_keys = ['header', 'phan1', 'phan2', 'phan3']

    for (p_crop, p_bin, p_cnt, bbox), key in zip(sections, section_keys):
        # Cắt vùng từ warped color
        crop_color = _crop_region(warped_color, bbox, padding=10)
        crop_gray = _crop_region(warped_gray, bbox, padding=10)

        # Ảnh cắt
        path_crop = os.path.join(debug_folder, f"{p_crop}_crop.jpg")
        _safe_imwrite(path_crop, crop_color)
        paths[f'{key}_crop'] = path_crop

        # Nhị phân (Otsu)
        crop_bin = _binarize_global(crop_gray)
        path_bin = os.path.join(debug_folder, f"{p_bin}_bin.jpg")
        _safe_imwrite(path_bin, crop_bin)
        paths[f'{key}_bin'] = path_bin

        # Contour trên vùng cắt
        crop_cnt = _draw_contours(crop_gray, crop_color)
        path_cnt = os.path.join(debug_folder, f"{p_cnt}_contour.jpg")
        _safe_imwrite(path_cnt, crop_cnt)
        paths[f'{key}_contour'] = path_cnt

    # ====== FINAL DEBUG (canvas với tất cả overlay) ======
    p = os.path.join(debug_folder, "99_final_debug.jpg")
    _safe_imwrite(p, final_canvas)
    paths['final_debug'] = p

    # Cũng lưu 1 bản final_debug ở cấp cha để giữ tương thích với code cũ
    legacy_path = os.path.join(out_dir, f"{image_name}_debug.jpg")
    _safe_imwrite(legacy_path, final_canvas)
    paths['legacy_debug'] = legacy_path

    # Ghi metadata nếu có
    if info is not None:
        meta_path = os.path.join(debug_folder, "_info.txt")
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write(f"image_name: {image_name}\n")
                f.write(f"align_method: {info.get('method', '?')}\n")
                f.write(f"skew_angle: {info.get('skew_angle', 0):.2f}°\n")
                f.write(f"rotated: {info.get('rotated', False)}\n")
                f.write(f"paper_found: {info.get('paper_found', False)}\n")
                f.write(f"scale: {info.get('scale', 1):.3f}\n")
                if 'src_pts' in info:
                    f.write("src_pts (TL, TR, BR, BL):\n")
                    for p in info['src_pts']:
                        f.write(f"  ({p[0]:.1f}, {p[1]:.1f})\n")
        except Exception:
            pass

    return paths
