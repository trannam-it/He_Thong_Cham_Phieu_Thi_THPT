# =============================================================================
# DETECTOR PHAN I - 40 cau trac nghiem ABCD (4 block x 10 cau)
# ---------------------------------------------------------------------------
# Quy trinh:
#   1. Lay 40 hang toa do bubble tu template (moi hang 4 bubble A/B/C/D).
#   2. Voi moi hang, score 4 bubble, chon bubble co score cao nhat + gap du lon.
#   3. Tra ve dict { 1..40 : 'A'/'B'/'C'/'D' }.
# =============================================================================

import cv2
from .template1 import phan1_bubble_centers, P1_BLOCKS
from .scoring import bubble_fill_ratio, pick_one


P1_FILL_THRESH = 0.45
P1_GAP_THRESH  = 0.18
P1_WEAK_THRESH = 0.28


class OMR_Phan1:
    """Detector Phan I: 40 cau trac nghiem ABCD."""

    LABELS = ['A', 'B', 'C', 'D']

    def process(self, warped_gray):
        """Cham phan 1 tu anh da warp.

        Returns:
            result (dict): { cau_no (int) : 'A'/'B'/'C'/'D' or None }
            certainty (dict): { cau_no : 'filled'/'weak'/'uncertain'/'empty' }
            debug (dict):
              'centers': { cau: [(cx,cy,r)..4] },
              'scores':  { cau: [s_A, s_B, s_C, s_D] }
        """
        print("\n" + "=" * 70)
        print("PHAN I: 40 CAU ABCD")
        print("=" * 70)

        centers = phan1_bubble_centers()
        result    = {}
        certainty = {}
        scores_all = {}

        for cau in sorted(centers.keys()):
            row = centers[cau]
            scores = [bubble_fill_ratio(warped_gray, cx, cy, r)[0]
                      for (cx, cy, r) in row]
            idx, cert = pick_one(scores,
                                  fill_thresh=P1_FILL_THRESH,
                                  gap_thresh=P1_GAP_THRESH,
                                  weak_thresh=P1_WEAK_THRESH)
            ans = self.LABELS[idx] if idx is not None else None
            result[cau]    = ans
            certainty[cau] = cert
            scores_all[cau] = scores
            print(f"  Cau {cau:2d}: {scores} -> {ans} ({cert})")

        debug = {'centers': centers, 'scores': scores_all}
        return result, certainty, debug

    # -----------------------------------------------------------------
    # VE DEBUG
    # -----------------------------------------------------------------
    def draw_debug(self, canvas, result, certainty, debug_data):
        """Ve overlay debug len anh warp."""
        for blk in P1_BLOCKS:
            cv2.rectangle(canvas, blk['bbox'][:2], blk['bbox'][2:],
                          (0, 165, 255), 2)

        centers = debug_data['centers']
        scores_all = debug_data['scores']
        for cau in sorted(centers.keys()):
            row = centers[cau]
            ans = result.get(cau)
            cert = certainty.get(cau, 'empty')
            chosen_idx = (self.LABELS.index(ans) if ans in self.LABELS else -1)

            # Nhan text ben trai bubble dau tien
            label_x = int(row[0][0]) - 50
            label_y = int(row[0][1]) + 4
            color = ((0, 0, 255) if cert == 'filled' else
                     (0, 140, 255) if cert in ('weak', 'uncertain') else
                     (0, 200, 0))
            cv2.putText(canvas, f"{cau}:{ans or '?'}",
                         (label_x, label_y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

            # Ve tung bubble
            for o_i, (cx, cy, r) in enumerate(row):
                is_chosen = (o_i == chosen_idx)
                if is_chosen and cert == 'filled':
                    c = (0, 0, 255); t = 2
                elif is_chosen and cert in ('weak', 'uncertain'):
                    c = (0, 140, 255); t = 2
                else:
                    c = (0, 200, 0); t = 1
                cv2.circle(canvas, (int(cx), int(cy)), int(r), c, t)
                # score %
                sc = scores_all[cau][o_i]
                cv2.putText(canvas, f"{int(sc*100)}",
                             (int(cx) - 10, int(cy) - int(r) - 3),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.30, c, 1)
