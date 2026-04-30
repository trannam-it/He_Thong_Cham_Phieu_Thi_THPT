# =============================================================================
# MAIN.PY - Giao dien he thong cham phieu trac nghiem THPT
# NANG CAP:
#   - Hien thi nhieu anh debug (cat, nhi phan, contour) cho tung phan
#   - Luu anh goc vao Anh_cham/Anh_chua_cham
#   - Tai file Excel ve bat ky vi tri nao nguoi dung chon
#   - Tien xu ly nang cao (deskew, CLAHE, ECC refine)
# =============================================================================

import sys
import os
import shutil
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QListWidget, QListWidgetItem, QScrollArea, QGroupBox, QTextEdit,
    QSplitter, QFrame, QAbstractItemView, QTabWidget, QGridLayout,
    QComboBox, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QFont, QIcon, QColor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
import omr_engine


# =============================================================================
# WORKER THREAD - Cham phieu chay nen
# =============================================================================

class GradingWorker(QThread):
    """Thread chay viec cham phieu de khong block giao dien."""
    progress = Signal(int, str)
    finished = Signal(list, str)
    error    = Signal(str)

    def __init__(self, image_paths, base_dir):
        super().__init__()
        self.image_paths = image_paths
        self.base_dir = base_dir

    def run(self):
        try:
            total = len(self.image_paths)
            debug_dir = os.path.join(self.base_dir, "Anh_cham", "Anh_da_cham")
            json_dir  = os.path.join(self.base_dir, "Diem", "JSON")
            xlsx_dir  = os.path.join(self.base_dir, "Diem", "XLSX")

            os.makedirs(debug_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)
            os.makedirs(xlsx_dir, exist_ok=True)

            all_results = []
            for idx, img_path in enumerate(self.image_paths):
                name = os.path.basename(img_path)
                self.progress.emit(
                    int((idx / total) * 100),
                    f"Dang cham {idx+1}/{total}: {name}"
                )

                try:
                    result = omr_engine.process_full_omr(
                        img_path,
                        output_debug_dir=debug_dir,
                        output_json_dir=json_dir,
                        base_dir=self.base_dir,
                        save_original=True,
                        export_debug_steps=True,
                    )
                    if result:
                        all_results.append(result)
                    else:
                        all_results.append({
                            'image_name': name,
                            'error': 'Khong xu ly duoc anh',
                        })
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        'image_name': name,
                        'error': str(e),
                    })

            self.progress.emit(95, "Dang tao file Excel...")
            from omr_engine.processor import _create_batch_excel
            xlsx_path = _create_batch_excel(all_results, xlsx_dir)

            self.progress.emit(100, "Hoan thanh!")
            self.finished.emit(all_results, xlsx_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# =============================================================================
# DEBUG VIEWER WIDGET - Tab hien thi cac anh debug (cat, nhi phan, contour)
# =============================================================================

# Cac anh debug duoc xuat boi debug_exporter, map theo ten key -> nhan hien thi
DEBUG_VIEW_LAYOUT = [
    # (group_title, [ (key, display_label) ])
    ("Toan anh",        [
        ('original',        '00. Anh goc'),
        ('warped',          '02. Warped (da nan chinh)'),
        ('binary_global',   '03. Nhi phan (Otsu)'),
        ('binary_adaptive', '04. Nhi phan (Adaptive)'),
        ('contours',        '05. Duong bien (contour)'),
        ('final_debug',     '99. Debug cuoi cung'),
    ]),
    ("Header (SBD + MDT)", [
        ('header_crop',    '10. Cat vung header'),
        ('header_bin',     '11. Nhi phan header'),
        ('header_contour', '12. Contour header'),
    ]),
    ("Phan I (40 ABCD)", [
        ('phan1_crop',    '20. Cat vung Phan I'),
        ('phan1_bin',     '21. Nhi phan Phan I'),
        ('phan1_contour', '22. Contour Phan I'),
    ]),
    ("Phan II (Dung/Sai)", [
        ('phan2_crop',    '30. Cat vung Phan II'),
        ('phan2_bin',     '31. Nhi phan Phan II'),
        ('phan2_contour', '32. Contour Phan II'),
    ]),
    ("Phan III (Dien so)", [
        ('phan3_crop',    '40. Cat vung Phan III'),
        ('phan3_bin',     '41. Nhi phan Phan III'),
        ('phan3_contour', '42. Contour Phan III'),
    ]),
]


class DebugImageViewer(QWidget):
    """Widget hien thi cac anh debug (cat tung phan, nhi phan, contour).

    Cau truc: combobox chon nhom -> combobox chon anh -> label hien thi.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_paths = {}   # dict key -> path
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Toolbar chon nhom va anh
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Nhom:"))
        self.combo_group = QComboBox()
        self.combo_group.setMinimumWidth(160)
        for title, _items in DEBUG_VIEW_LAYOUT:
            self.combo_group.addItem(title)
        self.combo_group.currentIndexChanged.connect(self._on_group_changed)
        bar.addWidget(self.combo_group)

        bar.addSpacing(12)
        bar.addWidget(QLabel("Anh:"))
        self.combo_image = QComboBox()
        self.combo_image.setMinimumWidth(220)
        self.combo_image.currentIndexChanged.connect(self._on_image_changed)
        bar.addWidget(self.combo_image)

        bar.addStretch(1)

        self.btn_open_folder = QPushButton("Mo thu muc debug")
        self.btn_open_folder.clicked.connect(self._open_debug_folder)
        bar.addWidget(self.btn_open_folder)

        layout.addLayout(bar)

        # Khu vuc hien thi anh
        self.image_label = QLabel("Chua co anh debug. Hay cham phieu truoc.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #bdc3c7;
                border-radius: 8px;
                background-color: #fafafa;
                color: #999;
                font-size: 13px;
                padding: 10px;
            }
        """)
        self.image_label.setMinimumSize(400, 400)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        scroll.setStyleSheet("QScrollArea { border: none; background: #fafafa; }")
        layout.addWidget(scroll, 1)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 10px; color: #666; padding: 2px;")
        layout.addWidget(self.info_label)

        # Khoi tao danh sach anh theo nhom mac dinh
        self._on_group_changed(0)

    def set_debug_paths(self, paths):
        """paths: dict key -> path do debug_exporter tra ve."""
        self.current_paths = paths or {}
        # Kich hoat lai combobox de refresh
        self._on_group_changed(self.combo_group.currentIndex())

    def clear(self):
        self.current_paths = {}
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Chua co anh debug. Hay cham phieu truoc.")
        self.info_label.setText("")

    def _on_group_changed(self, idx):
        self.combo_image.blockSignals(True)
        self.combo_image.clear()
        if 0 <= idx < len(DEBUG_VIEW_LAYOUT):
            _title, items = DEBUG_VIEW_LAYOUT[idx]
            for key, label in items:
                path = self.current_paths.get(key, "")
                exists = bool(path) and os.path.exists(path)
                # Them icon text cho biet co anh hay khong
                prefix = "[OK] " if exists else "[--] "
                self.combo_image.addItem(prefix + label, userData=key)
        self.combo_image.blockSignals(False)
        self._on_image_changed(0)

    def _on_image_changed(self, idx):
        if idx < 0:
            return
        key = self.combo_image.itemData(idx)
        if not key:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("(khong co)")
            self.info_label.setText("")
            return
        path = self.current_paths.get(key, "")
        if not path or not os.path.exists(path):
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f"Khong co anh '{key}'.\nHay cham lai phieu.")
            self.info_label.setText(f"key={key}")
            return

        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image_label.setText(f"Khong doc duoc: {path}")
            self.info_label.setText("")
            return

        # Fit vao khung scroll
        w = self.image_label.width() - 20
        h = self.image_label.height() - 20
        if w < 100: w = 700
        if h < 100: h = 700
        scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #fafafa;
                padding: 5px;
            }
        """)
        self.info_label.setText(
            f"{os.path.basename(path)} ({pixmap.width()}x{pixmap.height()})  |  {path}"
        )

    def _open_debug_folder(self):
        # Mo thu muc chua anh debug hien tai (neu co)
        for k in ('final_debug', 'warped', 'original'):
            p = self.current_paths.get(k)
            if p and os.path.exists(p):
                folder = os.path.dirname(p)
                _open_folder_in_os(folder)
                return
        # Fallback: mo Anh_da_cham
        folder = os.path.join(BASE_DIR, "Anh_cham", "Anh_da_cham")
        if os.path.exists(folder):
            _open_folder_in_os(folder)


def _open_folder_in_os(folder):
    """Mo folder bang file manager cua he dieu hanh."""
    import subprocess
    if sys.platform == 'win32':
        os.startfile(folder)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', folder])
    else:
        subprocess.Popen(['xdg-open', folder])


# =============================================================================
# MAIN WINDOW
# =============================================================================

class OMRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("He Thong Cham Phieu Trac Nghiem THPT")
        self.setMinimumSize(1300, 800)
        self.resize(1500, 900)

        self.image_paths = []
        self.last_results = []
        self.last_xlsx_path = None
        self.worker = None

        # Map: ten_anh -> dict debug_paths
        self.result_debug_paths = {}

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ============= TITLE =============
        title = QLabel("HE THONG CHAM PHIEU TRAC NGHIEM THPT")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1a5276; padding: 8px;")
        main_layout.addWidget(title)

        subtitle = QLabel("Phieu 3 phan: Phan I (40 cau ABCD) | Phan II (8 cau Dung/Sai) | Phan III (6 cau Dien so)")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #555; font-size: 11px; margin-bottom: 5px;")
        main_layout.addWidget(subtitle)

        # ============= TOOLBAR =============
        toolbar = QHBoxLayout()

        self.btn_load = QPushButton("  Tai anh len  ")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.setStyleSheet(self._button_style("#2980b9", "#3498db"))
        self.btn_load.clicked.connect(self.load_images)
        toolbar.addWidget(self.btn_load)

        self.btn_clear = QPushButton("  Xoa danh sach  ")
        self.btn_clear.setMinimumHeight(40)
        self.btn_clear.setStyleSheet(self._button_style("#7f8c8d", "#95a5a6"))
        self.btn_clear.clicked.connect(self.clear_images)
        toolbar.addWidget(self.btn_clear)

        self.btn_grade = QPushButton("  CHAM PHIEU  ")
        self.btn_grade.setMinimumHeight(40)
        self.btn_grade.setStyleSheet(self._button_style("#27ae60", "#2ecc71"))
        self.btn_grade.setEnabled(False)
        self.btn_grade.clicked.connect(self.start_grading)
        toolbar.addWidget(self.btn_grade)

        self.btn_download = QPushButton("  Tai Excel ket qua  ")
        self.btn_download.setMinimumHeight(40)
        self.btn_download.setStyleSheet(self._button_style("#e67e22", "#f39c12"))
        self.btn_download.setEnabled(False)
        self.btn_download.clicked.connect(self.download_excel)
        toolbar.addWidget(self.btn_download)

        self.btn_open_folder = QPushButton("  Mo thu muc ket qua  ")
        self.btn_open_folder.setMinimumHeight(40)
        self.btn_open_folder.setStyleSheet(self._button_style("#8e44ad", "#9b59b6"))
        self.btn_open_folder.clicked.connect(self.open_result_folder)
        toolbar.addWidget(self.btn_open_folder)

        main_layout.addLayout(toolbar)

        # ============= PROGRESS =============
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(24)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                background-color: #ecf0f1;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 5px;
            }
        """)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("San sang. Hay tai anh phieu thi len de cham.")
        self.status_label.setStyleSheet("color: #555; font-size: 12px; padding: 3px;")
        main_layout.addWidget(self.status_label)

        # ============= MAIN CONTENT (splitter 3 panels) =============
        splitter = QSplitter(Qt.Horizontal)

        # --- Left: Danh sach anh ---
        left_group = QGroupBox("Danh sach anh tai len")
        left_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        left_layout = QVBoxLayout(left_group)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #d5f5e3;
                color: #1a5276;
            }
        """)
        self.image_list.currentItemChanged.connect(self._on_image_selected)
        left_layout.addWidget(self.image_list)

        self.lbl_count = QLabel("0 anh")
        self.lbl_count.setAlignment(Qt.AlignCenter)
        self.lbl_count.setStyleSheet("font-size: 11px; color: #888;")
        left_layout.addWidget(self.lbl_count)

        splitter.addWidget(left_group)

        # --- Center: Tab preview / debug images ---
        center_group = QGroupBox("Xem truoc / Anh debug")
        center_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        center_layout = QVBoxLayout(center_group)

        self.tabs = QTabWidget()

        # Tab 1: Preview anh goc
        preview_tab = QWidget()
        preview_tab_layout = QVBoxLayout(preview_tab)
        preview_tab_layout.setContentsMargins(2, 2, 2, 2)

        self.preview_label = QLabel("Click vao ten anh de xem truoc")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #bdc3c7;
                border-radius: 8px;
                background-color: #fafafa;
                color: #999;
                font-size: 13px;
                padding: 10px;
            }
        """)
        self.preview_label.setMinimumSize(400, 450)

        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setWidget(self.preview_label)
        preview_scroll.setStyleSheet("""
            QScrollArea { border: none; background-color: #fafafa; }
        """)
        preview_tab_layout.addWidget(preview_scroll)

        self.preview_name_label = QLabel("")
        self.preview_name_label.setAlignment(Qt.AlignCenter)
        self.preview_name_label.setStyleSheet("font-size: 10px; color: #666; padding: 2px;")
        preview_tab_layout.addWidget(self.preview_name_label)

        self.tabs.addTab(preview_tab, "Xem truoc")

        # Tab 2: Debug images (cat, nhi phan, contour)
        self.debug_viewer = DebugImageViewer()
        self.tabs.addTab(self.debug_viewer, "Anh debug (cat/nhi phan/contour)")

        center_layout.addWidget(self.tabs)
        splitter.addWidget(center_group)

        # --- Right: Ket qua ---
        right_group = QGroupBox("Ket qua cham phieu")
        right_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        right_layout = QVBoxLayout(right_group)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 10))
        self.result_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #fdfefe;
            }
        """)
        self.result_text.setPlaceholderText(
            "Ket qua cham phieu se hien thi o day sau khi cham xong..."
        )
        right_layout.addWidget(self.result_text)

        splitter.addWidget(right_group)
        splitter.setSizes([220, 600, 500])

        main_layout.addWidget(splitter, stretch=1)

        # ============= FOOTER =============
        footer = QLabel(
            "Anh goc -> Anh_cham/Anh_chua_cham  |  "
            "Anh debug -> Anh_cham/Anh_da_cham  |  "
            "JSON -> Diem/JSON  |  Excel -> Diem/XLSX (co the tai ra bat ky vi tri)"
        )
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #999; font-size: 10px; padding: 5px;")
        main_layout.addWidget(footer)

    # -----------------------------------------------------------------
    # BUTTONS STYLE
    # -----------------------------------------------------------------
    def _button_style(self, bg, bg_hover):
        return f"""
            QPushButton {{
                background-color: {bg};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {bg_hover};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
                color: #7f8c8d;
            }}
            QPushButton:pressed {{
                background-color: {bg};
            }}
        """

    # -----------------------------------------------------------------
    # IMAGE PREVIEW
    # -----------------------------------------------------------------
    def _on_image_selected(self, current, previous):
        if current is None:
            return

        img_path = current.toolTip()
        if not img_path or not os.path.exists(img_path):
            self.preview_label.setText("Khong tim thay file anh")
            self.preview_name_label.setText("")
            self.debug_viewer.clear()
            return

        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            self.preview_label.setText("Khong the doc anh nay")
            self.preview_name_label.setText("")
            return

        preview_w = self.preview_label.width() - 20
        preview_h = self.preview_label.height() - 20
        if preview_w < 100:
            preview_w = 480
        if preview_h < 100:
            preview_h = 560

        scaled_pixmap = pixmap.scaled(
            preview_w, preview_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.preview_label.setPixmap(scaled_pixmap)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: #fafafa;
                padding: 5px;
            }
        """)

        name = os.path.basename(img_path)
        w_img = pixmap.width()
        h_img = pixmap.height()
        self.preview_name_label.setText(f"{name} ({w_img}x{h_img})")

        # Cap nhat tab debug neu anh nay da duoc cham
        debug_paths = self.result_debug_paths.get(name)
        if debug_paths:
            self.debug_viewer.set_debug_paths(debug_paths)
        else:
            self.debug_viewer.clear()

    # -----------------------------------------------------------------
    # LOAD IMAGES
    # -----------------------------------------------------------------
    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Chon anh phieu thi",
            "",
            "Anh (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Tat ca (*.*)"
        )
        if files:
            self.image_paths.extend(files)
            self.image_list.clear()
            for f in self.image_paths:
                item = QListWidgetItem(os.path.basename(f))
                item.setToolTip(f)
                self.image_list.addItem(item)

            count = len(self.image_paths)
            self.lbl_count.setText(f"{count} anh")
            self.btn_grade.setEnabled(count > 0)
            self.status_label.setText(f"Da tai {count} anh. Nhan 'CHAM PHIEU' de bat dau.")

            if count > 0:
                self.image_list.setCurrentRow(0)

    # -----------------------------------------------------------------
    # CLEAR
    # -----------------------------------------------------------------
    def clear_images(self):
        self.image_paths.clear()
        self.image_list.clear()
        self.result_debug_paths = {}
        self.lbl_count.setText("0 anh")
        self.btn_grade.setEnabled(False)
        self.status_label.setText("Da xoa danh sach. Hay tai anh moi.")

        self.preview_label.setText("Click vao ten anh de xem truoc")
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #bdc3c7;
                border-radius: 8px;
                background-color: #fafafa;
                color: #999;
                font-size: 13px;
                padding: 10px;
            }
        """)
        self.preview_name_label.setText("")
        self.debug_viewer.clear()

    # -----------------------------------------------------------------
    # START GRADING
    # -----------------------------------------------------------------
    def start_grading(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Khong co anh", "Hay tai anh len truoc!")
            return

        self.btn_load.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_grade.setEnabled(False)
        self.btn_download.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result_text.clear()
        self.result_text.append("Dang cham phieu...\n")

        self.worker = GradingWorker(self.image_paths, BASE_DIR)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_grading_finished)
        self.worker.error.connect(self._on_grading_error)
        self.worker.start()

    def _on_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_grading_finished(self, results, xlsx_path):
        self.last_results = results
        self.last_xlsx_path = xlsx_path

        # Luu map ten_anh -> debug_paths de khi chon anh hien debug
        self.result_debug_paths = {}
        for r in results:
            name = r.get('image_name')
            dp   = r.get('debug_paths', {})
            if name and dp:
                self.result_debug_paths[name] = dp

        self.btn_load.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_grade.setEnabled(True)
        self.btn_download.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.result_text.clear()
        self._display_results(results, xlsx_path)
        self.status_label.setText(
            f"Hoan thanh! Da cham {len(results)} anh. File Excel: {os.path.basename(xlsx_path)}"
        )

        # Chuyen sang tab debug cho anh dang chon (neu co)
        cur = self.image_list.currentItem()
        if cur is not None:
            self._on_image_selected(cur, None)
            # Neu co debug, tu dong mo tab debug de nguoi dung thay
            if self.result_debug_paths:
                self.tabs.setCurrentIndex(1)

    def _on_grading_error(self, error_msg):
        self.btn_load.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_grade.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"LOI: {error_msg}")
        QMessageBox.critical(self, "Loi cham phieu", f"Khong the cham phieu:\n{error_msg}")

    # -----------------------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------------------
    def _display_results(self, results, xlsx_path):
        txt = self.result_text

        txt.append("=" * 60)
        txt.append("         KET QUA CHAM PHIEU TRAC NGHIEM")
        txt.append("=" * 60)
        txt.append(f"So anh da cham: {len(results)}")
        txt.append(f"File Excel: {xlsx_path}")
        txt.append("")

        for idx, r in enumerate(results):
            txt.append("-" * 60)
            name = r.get('image_name', f'Anh {idx+1}')
            txt.append(f"[{idx+1}] {name}")

            if 'error' in r:
                txt.append(f"   LOI: {r['error']}")
                txt.append("")
                continue

            # Hien thong tin alignment
            align = r.get('align_method', '?')
            skew  = r.get('skew_angle', 0.0)
            txt.append(f"   Align:   {align}  (skew={skew:.2f} deg)")

            hdr = r.get('header', {})
            sbd = hdr.get('sbd', '?')
            madt = hdr.get('madt', '?')

            txt.append(f"   SBD:     {sbd}")
            txt.append(f"   Ma de:   {madt}")

            p1 = r.get('phan1', {})
            n_p1 = sum(1 for v in p1.values() if v is not None)
            txt.append(f"   Phan I (ABCD):   {n_p1}/40 cau co dap an")
            line = "     "
            for i in range(1, 41):
                val = p1.get(i, p1.get(str(i), None))
                line += f"{val if val else '-'} "
                if i % 10 == 0:
                    txt.append(line)
                    line = "     "

            p2 = r.get('phan2', {})
            n_p2 = sum(1 for q in p2.values()
                       for v in (q.values() if isinstance(q, dict) else [])
                       if v is not None)
            txt.append(f"   Phan II (Dung/Sai): {n_p2}/32 y co dap an")
            for q_num in sorted(p2.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
                q_data = p2[q_num]
                if isinstance(q_data, dict):
                    parts = []
                    for sub in ['a', 'b', 'c', 'd']:
                        val = q_data.get(sub, None)
                        short = 'D' if val == 'Dung' else ('S' if val == 'Sai' else '-')
                        parts.append(f"{sub}={short}")
                    txt.append(f"     Cau {q_num}: {' '.join(parts)}")

            p3 = r.get('phan3', {})
            n_p3 = sum(1 for v in p3.values() if v)
            txt.append(f"   Phan III (Dien so): {n_p3}/6 cau co dap an")
            for q_num in sorted(p3.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
                val = p3[q_num]
                txt.append(f"     Cau {q_num}: {val if val else '(trong)'}")

            txt.append("")

        txt.append("=" * 60)
        txt.append("HOAN THANH")
        txt.append("=" * 60)

    # -----------------------------------------------------------------
    # DOWNLOAD EXCEL - Luu duoc ra bat ky vi tri nguoi dung chon
    # -----------------------------------------------------------------
    def download_excel(self):
        """Mo hop thoai luu file Excel ket qua ra bat ky vi tri nao nguoi dung chon.

        Gia tri mac dinh la thu muc Documents hoac Home cua nguoi dung
        de thuan tien, thay vi trong project folder.
        """
        if not self.last_xlsx_path or not os.path.exists(self.last_xlsx_path):
            QMessageBox.warning(self, "Khong co ket qua",
                              "Chua co file ket qua. Hay cham phieu truoc!")
            return

        # Mac dinh goi y luu o Documents / Home cua nguoi dung
        home_dir = os.path.expanduser("~")
        docs_dir = os.path.join(home_dir, "Documents")
        default_dir = docs_dir if os.path.isdir(docs_dir) else home_dir

        # Ten file goi y dua theo timestamp
        base_name = os.path.basename(self.last_xlsx_path)
        default_path = os.path.join(default_dir, base_name)

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Luu file Excel ket qua (chon vi tri bat ky)",
            default_path,
            "Excel Files (*.xlsx);;All Files (*.*)"
        )

        if save_path:
            # Nguoi dung co the quen phan mo rong
            if not save_path.lower().endswith('.xlsx'):
                save_path += '.xlsx'
            try:
                shutil.copy2(self.last_xlsx_path, save_path)
                reply = QMessageBox.information(
                    self, "Thanh cong",
                    f"Da luu file ket qua tai:\n{save_path}\n\n"
                    f"Mo thu muc chua file?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    _open_folder_in_os(os.path.dirname(save_path))
            except Exception as e:
                QMessageBox.critical(
                    self, "Loi",
                    f"Khong the luu file:\n{str(e)}"
                )

    # -----------------------------------------------------------------
    # OPEN RESULT FOLDER
    # -----------------------------------------------------------------
    def open_result_folder(self):
        result_dir = os.path.join(BASE_DIR, "Diem")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        _open_folder_in_os(result_dir)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    from PySide6.QtGui import QPalette
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
    app.setPalette(palette)

    window = OMRApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
