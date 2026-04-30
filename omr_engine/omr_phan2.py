# =============================================================================
# DETECTOR PHAN II - 8 cau Dung/Sai (moi cau co 4 phan a,b,c,d)
# ---------------------------------------------------------------------------
# Quy trinh:
#   1. Lay 8 cau, moi cau co 4 label a/b/c/d, moi label 2 bubble (Dung, Sai).
#   2. Voi moi label: score 2 bubble, chon 1 trong 2.
#   3. Tra ve dict { 1..8 : { 'a'/'b'/'c'/'d' : 'Dung'/'Sai' or None } }
# =============================================================================

import cv2
from .template1 import phan2_bubble_centers, P2_BLOCKS
from .scoring import bubble_fill_ratio, pick_binary


P2_FILL_THRESH = 0.40
P2_GAP_THRESH  = 0.14
P2_WEAK_THRESH = 0.25


class OMR_Phan2:
    """Detector Phan II: 8 cau Dung/Sai."""

    def process(self, warped_gray):
        """Cham phan 2.

        Returns:
            result (dict): { cau (int) : {'a': 'Dung'/'Sai' or None, ...} }
            certainty: { cau : { label : 'filled'/... } }
            debug: { 'centers': phan2 grid, 'scores': { cau: { label: (s_D, s_S) } } }
        """
        print("\n" + "=" * 70)
        print("PHAN II: 8 CAU DUNG/SAI")
        print("=" * 70)

        centers = phan2_bubble_centers()
        result    = {}
        certainty = {}
        scores_all = {}

        for cau in sorted(centers.keys()):
            result[cau]    = {}
            certainty[cau] = {}
            scores_all[cau] = {}
            for lbl in ['a', 'b', 'c', 'd']:
                d_bub = centers[cau][lbl]['Dung']
                s_bub = centers[cau][lbl]['Sai']
                s_d, _ = bubble_fill_ratio(warped_gray, *d_bub)
                s_s, _ = bubble_fill_ratio(warped_gray, *s_bub)
                idx, cert = pick_binary(s_d, s_s,
                                         fill_thresh=P2_FILL_THRESH,
                                         gap_thresh=P2_GAP_THRESH,
                                         weak_thresh=P2_WEAK_THRESH)
                if idx == 0:
                    val = 'Dung'
                elif idx == 1:
                    val = 'Sai'
                else:
                    val = None
                result[cau][lbl]    = val
                certainty[cau][lbl] = cert
                scores_all[cau][lbl] = (s_d, s_s)

            print(f"  Cau {cau}: a={result[cau]['a']}, "
                  f"b={result[cau]['b']}, c={result[cau]['c']}, "
                  f"d={result[cau]['d']}")

        debug = {'centers': centers, 'scores': scores_all}
        return result, certainty, debug

    # -----------------------------------------------------------------
    # VE DEBUG
    # -----------------------------------------------------------------
    def draw_debug(self, canvas, result, certainty, debug_data):
        """Ve overlay debug phan II len anh warp."""
        for blk in P2_BLOCKS:
            cv2.rectangle(canvas, blk['bbox'][:2], blk['bbox'][2:],
                           (200, 100, 255), 2)

        centers = debug_data['centers']
        scores  = debug_data['scores']

        for cau in sorted(centers.keys()):
            for lbl in ['a', 'b', 'c', 'd']:
                val  = result.get(cau, {}).get(lbl)
                cert = certainty.get(cau, {}).get(lbl, 'empty')

                d_pos = centers[cau][lbl]['Dung']
                s_pos = centers[cau][lbl]['Sai']
                s_d, s_s = scores[cau][lbl]

                for bub_name, (cx, cy, r), sc in [
                    ('Dung', d_pos, s_d),
                    ('Sai',  s_pos, s_s),
                ]:
                    chosen = (val == bub_name)
                    if chosen and cert == 'filled':
                        c = (0, 0, 255); t = 2
                    elif chosen and cert in ('weak', 'uncertain'):
                        c = (0, 140, 255); t = 2
                    else:
                        c = (0, 200, 0); t = 1
                    cv2.circle(canvas, (int(cx), int(cy)), int(r), c, t)
                    cv2.putText(canvas, f"{int(sc*100)}",
                                 (int(cx) - 10, int(cy) - int(r) - 2),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.28, c, 1)
