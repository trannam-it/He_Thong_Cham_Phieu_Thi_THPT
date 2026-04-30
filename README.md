# 📋 Hệ Thống Chấm Phiếu Trắc Nghiệm Tự Động (OMR System)

---

## 🎯 Giới Thiệu Hệ Thống

Hệ thống **OMR (Optical Mark Recognition)** là một giải pháp tự động chấm phiếu trắc nghiệm cho các kỳ thi THPT. Hệ thống sử dụng xử lý ảnh nâng cao để nhận dạng và phân tích phiếu thi, cung cấp kết quả chính xác và nhanh chóng.

### Chức Năng Chính:

- 🖼️ **Nhận dạng hình ảnh**: Quét hoặc chụp ảnh phiếu thi
- 🔧 **Xử lý ảnh nâng cao**: Tự động căn chỉnh, chuẩn hóa, làm nổi bật câu trả lời
- 📊 **Chấm điểm tự động**: Nhận biết bubble đáp án và tính điểm
- 💾 **Xuất kết quả**: Lưu JSON từng phiếu và Excel tổng hợp
- 🔍 **Debug trực quan**: Hiển thị chi tiết các bước xử lý ảnh

---

## ⚡ Các Tính Năng Nổi Bật

### 1. 🔍 Debug Chi Tiết Đầy Đủ

Mỗi phiếu sau khi chấm sẽ tạo thư mục `<tên_ảnh>_debug/` chứa:

**Xử lý toàn ảnh:**

- `00_original.jpg` – Ảnh gốc đã đọc
- `02_warped.jpg` – Ảnh sau khi nắn chỉnh về khung 1100x1540
- `03_binary_global.jpg` – Ảnh nhị phân Otsu (toàn ảnh)
- `04_binary_adaptive.jpg` – Ảnh nhị phân adaptive (chống ánh sáng không đều)
- `05_contours.jpg` – Vẽ tất cả đường biên (contour)
- `99_final_debug.jpg` – Ảnh debug cuối cùng (vẽ bubble)

**Xử lý theo phần phiếu:**

- `10/11/12_header_*` – Header (SBD + Mã đề): cắt / nhị phân / contour
- `20/21/22_phan1_*` – Phần I (40 câu ABCD)
- `30/31/32_phan2_*` – Phần II (Câu Đúng/Sai)
- `40/41/42_phan3_*` – Phần III (Điền số)
- `_info.txt` – Metadata: phương pháp align, góc nghiêng, scale, 4 góc

### 2. 💾 Tự Động Lưu Ảnh Gốc

- Mỗi khi chấm, hệ thống tự động copy ảnh gốc vào `Anh_cham/Anh_chua_cham/`
- Tránh mất dữ liệu gốc của người dùng
- Nếu ảnh đã có thì bỏ qua (không ghi đè)

### 3. 🎨 Giao Diện Debug Trực Quan

App có **2 tab** ở giữa màn hình:

**Tab "Xem trước":**

- Hiển thị ảnh gốc ban đầu khi tải lên

**Tab "Ảnh debug":**

- **Chọn nhóm**: Toàn ảnh / Header / Phần I / Phần II / Phần III
- **Chọn ảnh**: Cắt vùng / Nhị phân / Contour của nhóm
- **Mở thư mục debug**: Xem trực tiếp file ngoài ứng dụng
- Tự động cập nhật khi chọn ảnh trong danh sách
- Sau khi chấm xong, app tự chuyển sang tab debug

### 4. 📊 Tải Excel Kết Quả Linh Hoạt

Nút **"Tải Excel kết quả"**:

- Mở hộp thoại lưu file
- Mặc định đến thư mục `~/Documents` thay vì project folder
- Hỗ trợ chọn **bất kỳ vị trí nào** trên máy
- Tự động thêm `.xlsx` nếu quên
- Hỏi người dùng có muốn mở thư mục vừa lưu

### 5. 📷 Ảnh Mẫu & Tinh Chỉnh Alignment (ECC)

Hệ thống tự động:

- Đọc ảnh đầu tiên trong `Anh_cham/Anh_mau_phieu/` làm ảnh mẫu
- Warp ảnh mẫu về kích thước chuẩn (1100x1540) và cache lại
- Dùng thuật toán **ECC (Enhanced Correlation Coefficient)** để tinh chỉnh alignment của từng phiếu
- Nếu ECC không hội tụ hoặc độ lệch > 50px, giữ nguyên kết quả warp cũ
- **Kết quả**: Alignment chính xác hơn, ít bỏ sót câu trả lời

### 6. 🖼️ Tiền Xử Lý Ảnh Nâng Cao

Xử lý ảnh mờ, nghiêng, tô không rõ bằng pipeline hiện đại:

**Bước 0: Nâng Cao Chất Lượng Ảnh**

- **CLAHE**: Tăng tương phản cục bộ, khắc phục ánh sáng không đều
- **Unsharp mask**: Làm rõ nét bằng trừ phiên bản blur
- **Non-Local Means Denoising**: Giảm nhiễu JPEG/camera

**Bước 1: Tự Động Nắn Thẳng (Deskew)**

- Phát hiện 4 góc tờ giấy bằng Canny + Contour
- Kiểm tra độ nghiêng so với trục ngang/dọc
- Chỉ xoay khi độ nghiêng > 2° (tránh xoay ảnh thẳng)
- Sử dụng perspective transform nắn tờ giấy

**Bước 2: Nhị Phân Kép (Otsu + Adaptive)**

- Kết hợp **Otsu + Adaptive threshold**
- Phát hiện marker tốt hơn trong điều kiện ánh sáng không đều

**Bước 3: Scoring 3 Chỉ Số**
Kết hợp 3 chỉ số để xác định bubble đã tô:

1. **fill_ratio**: Tỉ lệ pixel tối (ngưỡng 140)
2. **darkness**: Độ tối trung bình (1 - mean_gray/255)
3. **local_adapt**: So sánh tâm bubble với nền 4 góc

- Lấy `max` của 3 chỉ số → Robust với ảnh mờ/nhòe

**Bước 4: ECC Refine (Tinh Chỉnh)**

- Chạy `cv2.findTransformECC` sau khi warp
- Tinh chỉnh dựa trên ảnh mẫu
- Hữu ích khi marker phát hiện lệch vài pixel

---

## 📁 Cấu Trúc Thư Mục

```
He_Thong_ORM/
├── main.py                          # Giao diện PySide6 (ứng dụng chính)
├── requirements.txt                 # Thư viện cần cài đặt
├── requirements_full.txt            # Thư viện đầy đủ (phát triển)
├── README.md                        # File này
│
├── Anh_cham/
│   ├── Anh_chua_cham/               # 📥 Ảnh gốc (tự động lưu khi tải)
│   ├── Anh_da_cham/                 # 🔍 Debug output
│   │   └── <tên>_debug/             # Thư mục debug mỗi phiếu
│   │       ├── 00_original.jpg
│   │       ├── 02_warped.jpg
│   │       ├── 03_binary_global.jpg
│   │       └── ...
│   └── Anh_mau_phieu/               # 📐 Ảnh mẫu (dùng cho ECC refine)
│
├── Diem/
│   ├── JSON/                        # 📄 Kết quả JSON từng phiếu
│   └── XLSX/                        # 📊 File Excel tổng hợp
│
├── omr_engine/
│   ├── __init__.py
│   ├── alignment.py                 # ✅ CLAHE, deskew, ECC refine
│   ├── debug_exporter.py            # 🔧 Xuất debug images
│   ├── processor.py                 # ⚙️ Xử lý chính, gọi debug_exporter
│   ├── scoring.py                   # 📊 3-metric fill ratio, chấm bubble
│   ├── omr_header.py                # 👤 Xử lý Header (SBD, Mã đề)
│   ├── omr_phan1.py                 # ✏️ Xử lý Phần I (ABCD)
│   ├── omr_phan2.py                 # ✓ Xử lý Phần II (Đúng/Sai)
│   ├── omr_phan3.py                 # 🔢 Xử lý Phần III (Điền số)
│   ├── template.py                  # 📋 Quản lý template phiếu
│   └── constants.py                 # ⚙️ Các hằng số cấu hình
│
└── resources/
    └── img/                         # 🎨 Icon và tài nguyên UI
```

---

## 🚀 Cài Đặt & Chạy

### Yêu Cầu Hệ Thống

- **Python**: 3.8 trở lên
- **OS**: Windows, macOS, hoặc Linux
- **Ram**: Tối thiểu 4GB (8GB khuyên dùng)

### Cài Đặt

1. **Clone hoặc tải mã nguồn**

   ```bash
   cd He_Thong_OMR
   ```

2. **Cài đặt dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Chạy ứng dụng**
   ```bash
   python main.py
   ```

### Cài Đặt Phát Triển (đầy đủ)

```bash
pip install -r requirements_full.txt
```

---

## 📖 Hướng Dẫn Sử Dụng

### Quy Trình Cơ Bản

1. **Chuẩn Bị Ảnh Mẫu**
   - Đặt ít nhất 1 ảnh mẫu vào `Anh_cham/Anh_mau_phieu/`
   - Ảnh mẫu phải là phiếu chân phương (thẳng, rõ nét)
   - Ví dụ: `PhieuQG.0019.jpg`

2. **Mở Ứng Dụng**

   ```bash
   python main.py
   ```

3. **Tải Ảnh Phiếu**
   - Nhấn nút **"Tải ảnh lên"**
   - Chọn một hoặc nhiều ảnh phiếu thi
   - Ảnh sẽ hiển thị trong danh sách

4. **Chấm Phiếu**
   - Nhấn nút **"CHẤM PHIẾU"**
   - Tiến trình chạy nền (không block giao diện)
   - Có thể tải ảnh khác trong lúc chương trình đang chấm

5. **Xem Kết Quả**
   - **Tab "Xem trước"**: Xem ảnh gốc
   - **Tab "Ảnh debug"**: Xem chi tiết các bước xử lý
   - **Panel phải**: Xem kết quả chấm (SBD, Mã đề, Điểm từng phần)

6. **Xuất Kết Quả**
   - Nhấn **"Tải Excel kết quả"**
   - Chọn vị trí lưu file
   - File Excel chứa tất cả kết quả chấm của các phiếu đã xử lý

### Các Thao Tác Nâng Cao

**Xem Thông Tin Metadata:**

- File `_info.txt` trong thư mục debug chứa:
  - Phương pháp alignment (markers hay corners)
  - Góc nghiêng (deskew angle)
  - Scale factor
  - Tọa độ 4 góc tờ giấy

**Xóa Debug Cũ:**

- Xóa thư mục `Anh_cham/Anh_da_cham/*_debug/` nếu cần giải phóng dung lượng
- Mỗi phiếu debug chiếm ~18 ảnh (~3-5 MB)

---

## 🔧 Chi Tiết Kỹ Thuật

### Pipeline Xử Lý Mới (v3)

```
1. enhance_image_quality()
   └─ CLAHE + Unsharp + Denoise

2. detect_paper_corners()
   └─ Canny + Contour → tìm 4 góc tờ giấy

3. is_image_rotated(tol=2°)
   ├─ Thẳng (< 2°)  → KHÔNG xoay
   └─ Nghiêng (≥ 2°) → perspective transform

4. normalize_image()
   └─ Resize width về TEMPLATE_WIDTH giữ tỷ lệ

5. warp_to_template()
   └─ Resize về 1100x1540 bằng markers

6. refine_warp_by_reference()
   └─ ECC alignment với ảnh mẫu
```

### Các Hàm Chính

| File                | Hàm                             | Mục Đích                |
| ------------------- | ------------------------------- | ----------------------- |
| `alignment.py`      | `detect_paper_corners()`        | Phát hiện 4 góc tờ giấy |
| `alignment.py`      | `is_image_rotated()`            | Kiểm tra độ nghiêng     |
| `alignment.py`      | `warp_paper_to_rectangle()`     | Nắn tờ giấy             |
| `scoring.py`        | `calculate_bubble_fill_ratio()` | Xác định bubble tô      |
| `processor.py`      | `process_sheet()`               | Xử lý toàn bộ phiếu     |
| `debug_exporter.py` | `export_debug_images()`         | Xuất debug              |

---

## 🐛 Các Lỗi Được Sửa (v3)

### Lỗi 1: Ảnh Warped Bị Cắt Thay Vì Resize

**Triệu chứng:**

- Ảnh `02_warped.jpg` bị cắt một phần thay vì resize

**Nguyên nhân:**

- Khi marker detection thất bại, fallback trả về 4 góc ảnh
- Perspective transform không giữ tỷ lệ

**Giải pháp:**

- Thêm `detect_paper_corners()` tìm 4 góc tờ giấy thực tế
- Thêm `warp_paper_to_rectangle()` nắn về hình chữ nhật đúng tỷ lệ
- `warp_to_template()` chỉ resize (scale to fit, KHÔNG crop)

### Lỗi 2: Ảnh Thẳng Cũng Bị Xoay

**Triệu chứng:**

- Ảnh scan bằng scanner (góc ~0°) bị xoay sai lệch
- Kết quả lệch khỏi template

**Nguyên nhân:**

- `deskew_image()` cũ dùng Hough line, luôn xoay theo `median_angle`
- Hough line dễ cho ra góc nhỏ không mong muốn do bias của text

**Giải pháp:**

- Loại bỏ hoàn toàn `deskew_image()` cũ
- Dùng `is_image_rotated()` tính góc từ **CẠNH TỜ GIẤY**
- Chỉ xoay khi **max(góc 4 cạnh) > 2°**

### Lỗi 3: Contour Otsu Có Nhiễu Quá Nhiều

**Triệu chứng:**

- Ảnh `05_contours.jpg` có quá nhiều contour nhiễu
- Bubble trống không hiển thị

**Nguyên nhân:**

- `_draw_contours()` dùng Otsu binarization
- Otsu có nhiễu với ảnh ánh sáng không đều

**Giải pháp:**

- Đổi `_draw_contours()` dùng **Adaptive threshold** thay vì Otsu
- Adaptive threshold khử nhiễu tốt hơn
- Bubble trống và đã tô đều hiện rõ

---

## 📝 Ghi Chú & Lưu Ý

### Dung Lượng Ổ Đĩa

- Mỗi phiếu debug chiếm ~3-5 MB
- 1000 phiếu debug = ~3-5 GB
- Xóa thư mục `Anh_da_cham/*_debug/` khi không cần

### ECC Refine

- Có thể thất bại silently với ảnh quá khác ảnh mẫu
- Không phá vỡ pipeline, vẫn chấm bằng markers

### Lưu Trữ Metadata

- Góc nghiêng, phương pháp alignment, scale được lưu trong `_info.txt`
- Hữu ích cho kiểm tra chất lượng sau này

### Cách Bảo Trì

```bash
# Xóa debug cũ nếu cần
rm -rf Anh_cham/Anh_da_cham/*_debug/

# Xóa kết quả cũ (nếu muốn reset)
rm -rf Diem/JSON/*
rm -rf Diem/XLSX/*
```

---

## ⚙️ Cấu Hình & Tùy Chỉnh

### File Cấu Hình Chính

- `omr_engine/constants.py` – Các hằng số (kích thước template, ngưỡng, v.v.)

### Cấu Hình Template

Mở `constants.py` để điều chỉnh:

```python
TEMPLATE_WIDTH = 1100        # Chiều rộng chuẩn
TEMPLATE_HEIGHT = 1540       # Chiều cao chuẩn
DESKEW_THRESHOLD = 2.0       # Ngưỡng xoay (độ)
ECC_MAX_ITER = 5000          # Số lần lặp ECC
ECC_THRESHOLD = 0.0001       # Ngưỡng hội tụ ECC
```

---

## 🆘 Xử Lý Sự Cố

### Vấn Đề: Chấm Không Chính Xác

**Kiểm Tra:**

1. Ảnh mẫu có thẳng và rõ nét?
2. Ảnh gốc có bị lệch, mờ, hoặc tô không rõ?
3. Xem ảnh debug `05_contours.jpg` để kiểm tra nhận diện bubble

**Giải Pháp:**

- Thay ảnh mẫu khác (phải rõ nét, thẳng)
- Scan lại phiếu nếu quá mờ
- Điều chỉnh ngưỡng scoring trong `constants.py`

### Vấn Đề: Ứng Dụng Bị Treo

**Kiểm Tra:**

1. Xem terminal/console có lỗi Python không?
2. Ảnh có quá lớn (> 50 MB)?
3. RAM còn đủ không?

**Giải Pháp:**

- Compress ảnh trước khi tải lên
- Tăng RAM hoặc xóa debug cũ
- Kiểm tra log lỗi trong terminal

### Vấn Đề: Marker Detection Thất Bại

**Triệu chứng:**

- `_info.txt` hiển thị "phương pháp: corners"
- Ảnh `02_warped.jpg` bị cắt sai

**Giải Pháp:**

1. Kiểm tra ảnh gốc có rõ nét không (xem `00_original.jpg`)
2. Thay ảnh mẫu khác
3. Điều chỉnh `MARKER_MIN_AREA` trong `constants.py`

---

## 📚 Tài Liệu Tham Khảo

### Thư Viện Sử Dụng

- **OpenCV**: Xử lý ảnh (ảnh xây dựng)
- **PySide6**: Giao diện người dùng (GUI)
- **NumPy/SciPy**: Xử lý toán học
- **Pandas**: Xuất Excel

### Tài Liệu Bên Ngoài

- [OpenCV Docs](https://docs.opencv.org/)
- [PySide6 Docs](https://doc.qt.io/qt-6/)
- [NumPy Docs](https://numpy.org/doc/)

---

## 👨‍💼 Thông Tin Dự Án
 
**Ngôn Ngữ:** Python 3.8+  
**License:** MIT  
**Mục Đích:** Chấm tự động phiếu trắc nghiệm THPT  
**Trạng Thái:** Ổn định & Sản xuất

---

**Cảm ơn bạn đã sử dụng Hệ Thống OMR!** 🙏
