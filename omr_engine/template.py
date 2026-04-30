# =============================================================================
# TEMPLATE - Dinh nghia toan bo toa do chuan (canonical) cua mau phieu tra loi
# ---------------------------------------------------------------------------
# Y tuong: sau khi nan chinh anh (perspective warp) ve khung 1100x1540, toan bo
# toa do cua cac o bubble, cac bounding box cua tung phan deu duoc tinh truoc
# o day. Pipeline cham phieu chi viec doc gia tri pixel o cac toa do nay.
# ---------------------------------------------------------------------------
# Cac gia tri duoi day duoc CALIB bang cach do dac tren anh mau thuc te
# (PhieuQG.0145.jpg) sau khi chuan hoa chieu rong ve TEMPLATE_WIDTH.
# =============================================================================

# Kich thuoc anh canonical (template).
TEMPLATE_WIDTH  = 1100
TEMPLATE_HEIGHT = 1540

# Goc 4 diem tu cu anchor / fiducial markers chuan (trong he toa do template).
# Day la 4 marker goc (tren-trai, tren-phai, duoi-phai, duoi-trai) cua khung
# phieu - dung de tinh perspective warp.
TEMPLATE_CORNERS = [
    (  54,   92),  # Top-Left
    (1036,   92),  # Top-Right
    (1036, 1478),  # Bottom-Right
    (  54, 1478),  # Bottom-Left
]

# ---------------------------------------------------------------------------
# HEADER: So bao danh (SBD) + Ma de thi (MDT)
# SBD: 6 cot digit, moi cot 10 hang (0-9).  Khi to mot bubble o cot i, hang j,
#      ta hieu digit cua cot i = j.
# MDT: 3 cot digit, moi cot 10 hang (0-9).
# ---------------------------------------------------------------------------
# Toa do tam 6 cot x 10 hang cua SBD (calib truc tiep)
SBD_COL_X  = [792, 812, 832, 852, 872, 892]
SBD_ROW_Y  = [212, 242, 270, 300, 329, 358, 386, 416, 446, 474]
SBD_BUBBLE_R = 8.0

MDT_COL_X  = [950, 970, 990]
MDT_ROW_Y  = [212, 242, 270, 300, 328, 358, 386, 416, 444, 472]
MDT_BUBBLE_R = 8.0

SBD_N_COLS = len(SBD_COL_X)
SBD_N_ROWS = len(SBD_ROW_Y)
MDT_N_COLS = len(MDT_COL_X)
MDT_N_ROWS = len(MDT_ROW_Y)

# Bounding box lon cho SBD/MDT (dung de ve debug).
SBD_BBOX = (773, 207, 906, 496)
MDT_BBOX = (933, 207, 1007, 496)

# ---------------------------------------------------------------------------
# PHAN I: 40 cau trac nghiem ABCD, chia 4 block, moi block 10 cau.
# Toa do calib truc tiep tu anh mau.  Cot D cua block 3, 4 duoc noi suy.
# ---------------------------------------------------------------------------
# Moi block co 4 cot x (A,B,C,D) va 10 row y (cau 1..10, 11..20, ...)
# Ghi nho:
#   spacing cot ~ 41-42 px, spacing hang ~ 24-25 px
P1_BLOCKS = [
    {
        'bbox'    : (100, 569, 289, 820),
        'col_x'   : [136, 178, 218, 260],
        'row_y'   : [588, 612, 638, 662, 688, 712, 738, 762, 788, 812],
        'start_cau': 1,
    },
    {
        'bbox'    : (314, 569, 504, 820),
        'col_x'   : [370, 412, 452, 494],
        'row_y'   : [587, 612, 636, 662, 686, 712, 738, 762, 786, 812],
        'start_cau': 11,
    },
    {
        'bbox'    : (528, 569, 740, 820),    # widen to cover D column
        'col_x'   : [606, 646, 688, 730],    # D = 730 (interpolated from spacing 41-42)
        'row_y'   : [586, 612, 636, 662, 686, 710, 736, 762, 786, 812],
        'start_cau': 21,
    },
    {
        'bbox'    : (742, 569, 975, 820),    # widen to cover D column
        'col_x'   : [840, 882, 924, 966],    # D = 966 (interpolated)
        'row_y'   : [586, 612, 636, 662, 686, 710, 736, 762, 786, 810],
        'start_cau': 31,
    },
]
P1_BUBBLE_R = 10.0

# ---------------------------------------------------------------------------
# PHAN II: 8 cau Dung/Sai, chia 4 block, moi block 2 sub-block (2 cau).
#          Moi sub-block co 2 cot (Dung, Sai) va 4 hang (a, b, c, d).
# ---------------------------------------------------------------------------
# Calib tu anh mau:
#   Block 1 (cau 1,2): x cots [136, 176, 218, 260], y rows [938, 962, 988, 1012]
#   Block 2 (cau 3,4): x cots [368, 408, 452, 494]  (494 la noi suy, Hough bo qua)
#   Block 3 (cau 5,6): x cots [604, 643, 686, 728]  (noi suy cho cot Sai)
#   Block 4 (cau 7,8): x cots [840, 878, 920, 960]  (noi suy cho cot Sai)
#
# Cau trai thi co 2 cot dau Dung/Sai, cau phai co 2 cot cuoi Dung/Sai
P2_BLOCKS = [
    {
        'bbox'      : (115, 864, 289, 1022),
        'col_x_L'   : [136, 176],            # Cau 1: Dung=136, Sai=176
        'col_x_R'   : [218, 260],            # Cau 2: Dung=218, Sai=260
        'row_y'     : [938, 962, 988, 1012], # a, b, c, d
        'cau_L'     : 1, 'cau_R': 2,
    },
    {
        'bbox'      : (316, 864, 508, 1022),
        'col_x_L'   : [370, 410],
        'col_x_R'   : [452, 494],
        'row_y'     : [938, 962, 988, 1012],
        'cau_L'     : 3, 'cau_R': 4,
    },
    {
        'bbox'      : (526, 864, 742, 1022),
        'col_x_L'   : [606, 646],
        'col_x_R'   : [688, 730],
        'row_y'     : [936, 962, 988, 1012],
        'cau_L'     : 5, 'cau_R': 6,
    },
    {
        'bbox'      : (736, 864, 976, 1022),
        'col_x_L'   : [840, 878],
        'col_x_R'   : [922, 964],
        'row_y'     : [935, 962, 988, 1012],
        'cau_L'     : 7, 'cau_R': 8,
    },
]
P2_BUBBLE_R = 9.5

# ---------------------------------------------------------------------------
# PHAN III: 6 cau dien so, moi block co
#   - Header (4 o de viet so) - bo qua khi cham.
#   - 12 row bubble (-, ',', 0..9) x 4 cot.
#     Row 0 = '-'  (chi co 1 bubble o cot 0, 3 cot con lai la trong)
#     Row 1 = ','  (chi co 3 bubble o cot 1,2,3, cot 0 trong)
#     Row 2..11 = digit 0..9 (du 4 bubble moi hang)
# ---------------------------------------------------------------------------
P3_BLOCKS = [
    {
        'bbox'  : (101, 1099, 246, 1458),
        'col_x' : [132, 156, 182, 206],
        'row_y' : [1156, 1180, 1206, 1230, 1254, 1280, 1304, 1330, 1354, 1378, 1404, 1428],
        'cau'   : 1,
    },
    {
        'bbox'  : (247, 1099, 386, 1458),
        'col_x' : [282, 306, 332, 356],
        'row_y' : [1154, 1180, 1205, 1230, 1255, 1278, 1304, 1328, 1352, 1378, 1402, 1428],
        'cau'   : 2,
    },
    {
        'bbox'  : (388, 1099, 527, 1458),
        'col_x' : [430, 456, 480, 506],
        'row_y' : [1154, 1179, 1204, 1230, 1254, 1279, 1302, 1328, 1352, 1376, 1402, 1427],
        'cau'   : 3,
    },
    {
        'bbox'  : (529, 1099, 667, 1458),
        'col_x' : [580, 606, 630, 655],
        'row_y' : [1152, 1178, 1204, 1229, 1254, 1278, 1302, 1328, 1352, 1376, 1402, 1426],
        'cau'   : 4,
    },
    {
        'bbox'  : (669, 1099, 807, 1458),
        'col_x' : [730, 756, 780, 806],
        'row_y' : [1152, 1178, 1202, 1228, 1254, 1278, 1302, 1326, 1352, 1376, 1400, 1426],
        'cau'   : 5,
    },
    {
        'bbox'  : (809, 1099, 948, 1458),
        'col_x' : [880, 906, 930, 955],    # 955 la noi suy cho cot 4
        'row_y' : [1152, 1177, 1202, 1228, 1252, 1278, 1302, 1326, 1350, 1376, 1400, 1426],
        'cau'   : 6,
    },
]
P3_BUBBLE_R  = 10.0
P3_N_ROWS    = 12
P3_ROW_LABELS = ['-', ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Chi so row 0 = '-' chi co o cot dau (col 0)
# Chi so row 1 = ',' chi co o cot 1, 2, 3
P3_ROW0_VALID_COLS = [0]               # '-' o cot 0
P3_ROW1_VALID_COLS = [1, 2, 3]         # ',' o cot 1,2,3

# ---------------------------------------------------------------------------
# FIDUCIAL DETECTION - Cac marker den nho tren canh trai/phai de nan chinh.
# ---------------------------------------------------------------------------
FIDUCIAL_MIN_W  = 14
FIDUCIAL_MAX_W  = 35
FIDUCIAL_MIN_H  = 4
FIDUCIAL_MAX_H  = 16
FIDUCIAL_MIN_AR = 1.7
FIDUCIAL_MAX_AR = 9.0
FIDUCIAL_EDGE_MARGIN_RATIO = 0.08

# Toa do tam cua 4 marker goc (TL, TR, BR, BL)
TEMPLATE_FIDUCIAL_TL = ( 54,  92)
TEMPLATE_FIDUCIAL_TR = (1036, 92)
TEMPLATE_FIDUCIAL_BR = (1036, 1478)
TEMPLATE_FIDUCIAL_BL = ( 54, 1478)


# =============================================================================
# HAM HO TRO - TAO DANH SACH TOA DO BUBBLE
# =============================================================================

def header_bubble_centers():
    """Tra ve 2 tuple (sbd, mdt), moi tuple la list-of-list (cx, cy, r).

    sbd[col][row] = (cx, cy, r). Co SBD_N_COLS cot x SBD_N_ROWS hang.
    """
    sbd = []
    for cx in SBD_COL_X:
        col = []
        for cy in SBD_ROW_Y:
            col.append((float(cx), float(cy), SBD_BUBBLE_R))
        sbd.append(col)
    mdt = []
    for cx in MDT_COL_X:
        col = []
        for cy in MDT_ROW_Y:
            col.append((float(cx), float(cy), MDT_BUBBLE_R))
        mdt.append(col)
    return sbd, mdt


def phan1_bubble_centers():
    """Tra ve dict { cau_no (int): [(cx, cy, r) for A, B, C, D] }."""
    centers = {}
    for blk in P1_BLOCKS:
        for r_i, cy in enumerate(blk['row_y']):
            cau = blk['start_cau'] + r_i
            row_bubs = []
            for cx in blk['col_x']:
                row_bubs.append((float(cx), float(cy), P1_BUBBLE_R))
            centers[cau] = row_bubs
    return centers


def phan2_bubble_centers():
    """Tra ve dict { cau_no : { label: {'Dung':(cx,cy,r), 'Sai':(cx,cy,r)} } }.

    8 cau, moi cau 4 label a,b,c,d, moi label 2 bubble.
    """
    centers = {}
    for blk in P2_BLOCKS:
        for side, (cau, col_x) in [('L', (blk['cau_L'], blk['col_x_L'])),
                                    ('R', (blk['cau_R'], blk['col_x_R']))]:
            centers[cau] = {}
            for r_i, label in enumerate(['a', 'b', 'c', 'd']):
                cy = blk['row_y'][r_i]
                centers[cau][label] = {
                    'Dung': (float(col_x[0]), float(cy), P2_BUBBLE_R),
                    'Sai':  (float(col_x[1]), float(cy), P2_BUBBLE_R),
                }
    return centers


def phan3_bubble_centers():
    """Tra ve dict { cau_no : { col_idx (0..3): [(cx,cy,r) for 12 row] } }.

    col_idx la vi tri digit tu trai sang phai (0 = digit cao nhat,
    3 = digit thap nhat). Row 0 la '-', row 1 la ',', row 2..11 la 0..9.
    """
    centers = {}
    for blk in P3_BLOCKS:
        cau = blk['cau']
        centers[cau] = {}
        for col_i, cx in enumerate(blk['col_x']):
            col_points = []
            for cy in blk['row_y']:
                col_points.append((float(cx), float(cy), P3_BUBBLE_R))
            centers[cau][col_i] = col_points
    return centers


def phan3_is_valid_cell(row_idx, col_idx):
    """Cho biet o (row, col) trong Phan III co bubble hay khong.

    Row 0 = '-' chi co o col 0.
    Row 1 = ',' chi co o col 1, 2, 3.
    Row >= 2 (digit 0..9) du 4 cot.
    """
    if row_idx == 0:
        return col_idx in P3_ROW0_VALID_COLS
    if row_idx == 1:
        return col_idx in P3_ROW1_VALID_COLS
    return True
