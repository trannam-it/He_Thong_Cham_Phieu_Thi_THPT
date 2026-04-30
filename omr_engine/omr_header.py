# =============================================================================
# DETECTOR HEADER (SBD + Ma de thi)
# ---------------------------------------------------------------------------
# Quy trinh:
#   1. Doc toa do tam 6 cot SBD + 3 cot MDT tu template.
#   2. Voi moi cot: score 10 bubble (hang 0..9), chon bubble co score cao nhat.
#   3. Ghep digits lai tao thanh SBD + MDT.
# =============================================================================

import cv2
import numpy as np
from .template1 import (
    header_bubble_centers,
    SBD_BBOX, MDT_BBOX, SBD_N_COLS, MDT_N_COLS,
)
from .scoring import bubble_fill_ratio, pick_one


# Nguong danh gia bubble cho header
HEADER_FILL_THRESH = 0.45
HEADER_GAP_THRESH  = 0.18
HEADER_WEAK_THRESH = 0.26


class OMR_Header:
    """Detector phan header (SBD + Ma de thi) tren anh da nan chinh."""

    def process(self, warped_gray):
        """Nhan dien header tu anh da warp.

        Args:
            warped_gray: anh grayscale da nan chinh theo template.

        Returns:
            result (dict), debug_data (dict).
        """
        print("\n" + "=" * 70)
        print("HEADER: SO BAO DANH + MA DE THI")
        print("=" * 70)

        sbd_grid, mdt_grid = header_bubble_centers()

        sbd_digits = []
        sbd_certainty = []
        sbd_scores = []   # [col][row] = score

        for col_i, col in enumerate(sbd_grid):
            scores = [bubble_fill_ratio(warped_gray, cx, cy, r)[0]
                      for (cx, cy, r) in col]
            idx, cert = pick_one(scores,
                                  fill_thresh=HEADER_FILL_THRESH,
                                  gap_thresh=HEADER_GAP_THRESH,
                                  weak_thresh=HEADER_WEAK_THRESH)
            sbd_digits.append(idx)
            sbd_certainty.append(cert)
            sbd_scores.append(scores)
            print(f"  SBD col {col_i}: digit={idx} cert={cert} "
                  f"max={max(scores):.3f}")

        mdt_digits = []
        mdt_certainty = []
        mdt_scores = []

        for col_i, col in enumerate(mdt_grid):
            scores = [bubble_fill_ratio(warped_gray, cx, cy, r)[0]
                      for (cx, cy, r) in col]
            idx, cert = pick_one(scores,
                                  fill_thresh=HEADER_FILL_THRESH,
                                  gap_thresh=HEADER_GAP_THRESH,
                                  weak_thresh=HEADER_WEAK_THRESH)
            mdt_digits.append(idx)
            mdt_certainty.append(cert)
            mdt_scores.append(scores)
            print(f"  MDT col {col_i}: digit={idx} cert={cert} "
                  f"max={max(scores):.3f}")

        sbd_str  = ''.join(str(d) if d is not None else '?' for d in sbd_digits)
        mdt_str  = ''.join(str(d) if d is not None else '?' for d in mdt_digits)

        print(f"\n  SBD = {sbd_str}")
        print(f"  MDT = {mdt_str}")

        result = {
            'sbd':            sbd_str,
            'madt':           mdt_str,
            'sbd_digits':     sbd_digits,
            'madt_digits':    mdt_digits,
            'sbd_certainty':  sbd_certainty,
            'madt_certainty': mdt_certainty,
        }
        debug_data = {
            'sbd_grid':   sbd_grid,
            'mdt_grid':   mdt_grid,
            'sbd_scores': sbd_scores,
            'mdt_scores': mdt_scores,
        }
        return result, debug_data

    # -----------------------------------------------------------------
    # VE DEBUG
    # -----------------------------------------------------------------
    def draw_debug(self, canvas, result, debug_data):
        """Ve overlay de bug len anh da warp (canvas)."""
        cv2.rectangle(canvas, SBD_BBOX[:2], SBD_BBOX[2:], (200, 100, 0), 2)
        cv2.rectangle(canvas, MDT_BBOX[:2], MDT_BBOX[2:], (200, 100, 0), 2)

        cv2.putText(canvas, f"SBD: {result['sbd']}",
                     (SBD_BBOX[0], SBD_BBOX[1] - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
        cv2.putText(canvas, f"MDT: {result['madt']}",
                     (MDT_BBOX[0], MDT_BBOX[1] - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)

        for block_name, grid, scores_all, chosen, cert_list in [
            ('sbd', debug_data['sbd_grid'], debug_data['sbd_scores'],
             result['sbd_digits'], result['sbd_certainty']),
            ('mdt', debug_data['mdt_grid'], debug_data['mdt_scores'],
             result['madt_digits'], result['madt_certainty']),
        ]:
            for col_i, col in enumerate(grid):
                digit = chosen[col_i]
                cert  = cert_list[col_i]
                for row_i, (cx, cy, r) in enumerate(col):
                    is_chosen = (digit is not None and row_i == digit)
                    if is_chosen and cert == 'filled':
                        color = (0, 0, 255); thick = 2
                    elif is_chosen and cert in ('weak', 'uncertain'):
                        color = (0, 140, 255); thick = 2
                    else:
                        color = (0, 200, 0); thick = 1
                    cv2.circle(canvas, (int(cx), int(cy)), int(r),
                                color, thick)
