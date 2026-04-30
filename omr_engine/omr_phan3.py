# =============================================================================
# DETECTOR PHAN III - 6 cau dien so (mot digit / cot, 4 cot / cau)
# ---------------------------------------------------------------------------
# Moi block co 4 cot digit, moi cot co 12 row:
#   row 0  = '-'  (chi co 1 bubble o cot 0)
#   row 1  = ','  (chi co 3 bubble o cot 1, 2, 3)
#   row 2  = 0
#   ...
#   row 11 = 9
#
# Quy trinh:
#   1. Voi moi cau (1..6), duyet 4 cot:
#      - Score 12 bubble (nhung bo qua o khong hop le).
#      - Chon row co score cao nhat + gap du lon -> cho ra ky tu cua cot do.
#   2. Ghep 4 ky tu lai thanh dap an cua cau (vd "1,23", "-12", "0,5", ...).
# =============================================================================

import cv2
from .template1 import (
    phan3_bubble_centers, phan3_is_valid_cell,
    P3_BLOCKS, P3_ROW_LABELS,
)
from .scoring import bubble_fill_ratio, pick_one


P3_FILL_THRESH = 0.45
P3_GAP_THRESH  = 0.18
P3_WEAK_THRESH = 0.28


class OMR_Phan3:
    """Detector Phan III: 6 cau dien so bang to bubble."""

    def process(self, warped_gray):
        """Cham phan III.

        Returns:
            result (dict): { cau (int) : 'dap_an' (string) }
            certainty: { cau : { col_i : 'filled'/... } }
            debug: { 'centers': grid, 'scores': { cau: { col: [..12] } } }
        """
        print("\n" + "=" * 70)
        print("PHAN III: 6 CAU DIEN SO")
        print("=" * 70)

        centers = phan3_bubble_centers()
        result    = {}
        certainty = {}
        scores_all = {}
        chosen_row = {}

        for cau in sorted(centers.keys()):
            cols = centers[cau]
            col_chars = []
            certainty[cau] = {}
            scores_all[cau] = {}
            chosen_row[cau] = {}
            for col_i, points in cols.items():
                scores = []
                valid_idxs = []
                for r_i, (cx, cy, r) in enumerate(points):
                    if phan3_is_valid_cell(r_i, col_i):
                        s, _ = bubble_fill_ratio(warped_gray, cx, cy, r)
                        scores.append(s)
                        valid_idxs.append(r_i)
                    else:
                        scores.append(-1.0)    # placeholder

                valid_scores = [scores[i] for i in valid_idxs]
                idx_rel, cert = pick_one(valid_scores,
                                          fill_thresh=P3_FILL_THRESH,
                                          gap_thresh=P3_GAP_THRESH,
                                          weak_thresh=P3_WEAK_THRESH)
                certainty[cau][col_i] = cert
                scores_all[cau][col_i] = scores

                if idx_rel is None:
                    col_chars.append('')
                    chosen_row[cau][col_i] = None
                else:
                    real_row = valid_idxs[idx_rel]
                    col_chars.append(P3_ROW_LABELS[real_row])
                    chosen_row[cau][col_i] = real_row

            ans = self._format_answer(col_chars)
            result[cau] = ans
            print(f"  Cau {cau}: cols={col_chars} -> '{ans}'")

        debug = {
            'centers':     centers,
            'scores':      scores_all,
            'chosen_row':  chosen_row,
        }
        return result, certainty, debug

    # -----------------------------------------------------------------
    # HAM LIEN QUAN DAP AN PHAN III
    # -----------------------------------------------------------------
    def _format_answer(self, col_chars):
        """Ghep 4 ky tu tu 4 cot thanh dap an.

        Quy tac:
          - Bo cac o rong o DAU va CUOI.
          - Giua chuoi, neu gap 1 o rong thi cung bo.
          - Neu co ky tu '-' -> phai o dau.
          - Neu co ky tu ',' -> dau phay phai nam dung 1 vi tri trong so 4.
        """
        # Buoc 1: noi lai bang text, xoa cac ky tu ''
        s = ''.join([c for c in col_chars if c != ''])

        # Sua '-' neu khong o dau
        if '-' in s:
            # Chi giu dau '-' o vi tri dau tien
            pos = s.index('-')
            if pos > 0:
                s = s.replace('-', '')    # bo dau - neu khong o dau
            else:
                # dau '-' o dau, con '-' khac phai bi loai
                s = '-' + s[1:].replace('-', '')

        # Sua ',' - chi giu 1 dau ,
        if s.count(',') > 1:
            first = s.index(',')
            s = s[:first+1] + s[first+1:].replace(',', '')

        return s

    # -----------------------------------------------------------------
    # VE DEBUG
    # -----------------------------------------------------------------
    def draw_debug(self, canvas, result, certainty, debug_data):
        """Ve overlay debug phan III len anh warp."""
        for blk in P3_BLOCKS:
            cv2.rectangle(canvas, blk['bbox'][:2], blk['bbox'][2:],
                          (255, 100, 100), 2)

        centers = debug_data['centers']
        scores_all = debug_data['scores']
        chosen_row = debug_data['chosen_row']

        for cau in sorted(centers.keys()):
            cols = centers[cau]
            # Ve dap an phia tren block
            blk_bbox = P3_BLOCKS[cau - 1]['bbox']
            cv2.putText(canvas, f"Cau {cau}: {result[cau]}",
                         (blk_bbox[0], blk_bbox[1] - 6),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 200), 2)

            for col_i, points in cols.items():
                cert = certainty[cau].get(col_i, 'empty')
                real_row = chosen_row[cau].get(col_i)
                for r_i, (cx, cy, r) in enumerate(points):
                    if not phan3_is_valid_cell(r_i, col_i):
                        continue
                    is_chosen = (real_row is not None and r_i == real_row)
                    if is_chosen and cert == 'filled':
                        c = (0, 0, 255); t = 2
                    elif is_chosen and cert in ('weak', 'uncertain'):
                        c = (0, 140, 255); t = 2
                    else:
                        c = (0, 200, 0); t = 1
                    cv2.circle(canvas, (int(cx), int(cy)), int(r), c, t)
