# =============================================================================
# OMR Engine - He thong cham phieu trac nghiem THPT
# ---------------------------------------------------------------------------
# Pipeline: Template + Perspective Warp + ECC refine + Detector cho tung phan.
# Ho tro phieu 3 phan:
#   - Phan I   : 40 cau trac nghiem ABCD.
#   - Phan II  : 8 cau Dung/Sai (moi cau 4 y a,b,c,d).
#   - Phan III : 6 cau dien so bang to bubble (-, ',', 0..9).
# =============================================================================

from .processor import process_full_omr, process_batch
from .debug_exporter import export_debug_images

__all__ = [
    'process_full_omr',
    'process_batch',
    'export_debug_images',
]
