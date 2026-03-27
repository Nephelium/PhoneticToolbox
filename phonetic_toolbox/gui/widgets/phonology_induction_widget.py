from __future__ import annotations

import os

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from phonetic_toolbox.services.phonology_service import PhonologyInductionService
from phonetic_toolbox.utils import get_resource_path


class ToneOrderTableWidget(QTableWidget):
    def __init__(self, rows: int, columns: int, parent=None):
        super().__init__(rows, columns, parent)
        self._drag_source_row = -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_source_row = self.rowAt(event.position().toPoint().y())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_source_row >= 0:
            target_row = self.rowAt(event.position().toPoint().y())
            if target_row >= 0 and target_row != self._drag_source_row:
                self._swap_rows(self._drag_source_row, target_row)
            self._drag_source_row = -1
        super().mouseReleaseEvent(event)

    def dropEvent(self, event):
        event.ignore()

    def _swap_rows(self, source_row: int, target_row: int):
        source_tone_text = self.item(source_row, 0).text() if self.item(source_row, 0) else ""
        source_class_text = self.item(source_row, 1).text() if self.item(source_row, 1) else ""
        target_tone_text = self.item(target_row, 0).text() if self.item(target_row, 0) else ""
        target_class_text = self.item(target_row, 1).text() if self.item(target_row, 1) else ""

        self._set_row_values(source_row, target_tone_text, target_class_text)
        self._set_row_values(target_row, source_tone_text, source_class_text)
        self.setCurrentCell(target_row, 0)

    def _set_row_values(self, row: int, tone_value: str, class_value: str):
        tone_item = QTableWidgetItem(tone_value)
        tone_item.setFlags(tone_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        tone_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        class_item = QTableWidgetItem(class_value)
        class_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(row, 0, tone_item)
        self.setItem(row, 1, class_item)


class ToneClassDialog(QDialog):
    def __init__(self, tones: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("调值归并设置")
        self.resize(420, 320)
        layout = QVBoxLayout(self)
        tip = QLabel("请拖拽行调整顺序（该顺序即文档展示顺序），并填写调类名称。")
        tip2 = QLabel("操作说明：按住某一行左键拖到目标行后松开，即与目标行互换。")
        tip2.setStyleSheet("color: #888888;")
        layout.addWidget(tip)
        layout.addWidget(tip2)
        self.table = ToneOrderTableWidget(len(tones), 2)
        self.table.setHorizontalHeaderLabels(["调值", "调类"])
        self.table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setDragEnabled(False)
        self.table.setDropIndicatorShown(False)
        self.table.setDragDropMode(QTableWidget.DragDropMode.NoDragDrop)
        for row_idx, tone in enumerate(tones):
            tone_item = QTableWidgetItem(tone)
            tone_item.setFlags(tone_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            tone_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 0, tone_item)
            tone_class_item = QTableWidgetItem(tone)
            tone_class_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 1, tone_class_item)
        self.table.horizontalHeaderItem(0).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.horizontalHeaderItem(1).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setColumnWidth(0, 88)
        self.table.setColumnWidth(1, 116)
        self.table.horizontalHeader().setStretchLastSection(False)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_btn = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_btn:
            ok_btn.setText("确定")
        if cancel_btn:
            cancel_btn.setText("取消")
        layout.addWidget(buttons)

    def collect_mapping(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for row_idx in range(self.table.rowCount()):
            tone = self.table.item(row_idx, 0).text().strip()
            tone_class = self.table.item(row_idx, 1).text().strip()
            mapping[tone] = tone_class or tone
        return mapping

    def collect_tone_order(self) -> list[str]:
        ordered: list[str] = []
        for row_idx in range(self.table.rowCount()):
            item = self.table.item(row_idx, 0)
            if item is None:
                continue
            tone = item.text().strip()
            if tone:
                ordered.append(tone)
        return ordered


class SymbolOrderDialog(QDialog):
    def __init__(self, initials: list[str], finals: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("声母/韵母排序")
        self.resize(620, 420)
        layout = QVBoxLayout(self)
        tip = QLabel(
            "请拖拽排序：该顺序将用于声母表、韵母表和同音字表。\n"
            "快捷键：Ctrl 可多选离散项；Shift 可批量选中首尾及中间项。"
        )
        layout.addWidget(tip)

        lists_layout = QHBoxLayout()
        self.initial_list = QListWidget()
        self.final_list = QListWidget()
        self._setup_sortable_list(self.initial_list)
        self._setup_sortable_list(self.final_list)
        self.initial_merge_map: dict[str, str] = {}
        self.final_merge_map: dict[str, str] = {}
        self.pending_merge: tuple[str, str] | None = None

        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("声母顺序"))
        left_panel.addWidget(self.initial_list)
        self.btn_merge_initial = QPushButton("归并声母")
        left_panel.addWidget(self.btn_merge_initial)
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("韵母顺序"))
        right_panel.addWidget(self.final_list)
        self.btn_merge_final = QPushButton("归并韵母")
        right_panel.addWidget(self.btn_merge_final)
        lists_layout.addLayout(left_panel)
        lists_layout.addLayout(right_panel)
        layout.addLayout(lists_layout)
        self.merge_tip = QLabel("归并：先单击要归并的音标，再点“归并”按钮，最后点击目标音标。")
        layout.addWidget(self.merge_tip)

        for value in initials:
            self._append_symbol_item(self.initial_list, value)
        for value in finals:
            self._append_symbol_item(self.final_list, value)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_btn = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_btn:
            ok_btn.setText("确定")
        if cancel_btn:
            cancel_btn.setText("取消")
        layout.addWidget(buttons)
        self.btn_merge_initial.clicked.connect(
            lambda: self._prepare_merge(self.initial_list, "initial")
        )
        self.btn_merge_final.clicked.connect(
            lambda: self._prepare_merge(self.final_list, "final")
        )
        self.initial_list.itemClicked.connect(
            lambda item: self._try_finish_merge(item, "initial")
        )
        self.final_list.itemClicked.connect(
            lambda item: self._try_finish_merge(item, "final")
        )

    def _setup_sortable_list(self, widget: QListWidget):
        widget.setDragEnabled(True)
        widget.viewport().setAcceptDrops(True)
        widget.setAcceptDrops(True)
        widget.setDropIndicatorShown(True)
        widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        widget.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

    def _append_symbol_item(self, widget: QListWidget, value: str):
        label = value if value else "空韵"
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, value)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        widget.addItem(item)

    def collect_orders(self) -> tuple[list[str], list[str], dict[str, str], dict[str, str]]:
        initials = [
            self.initial_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.initial_list.count())
        ]
        finals = [
            self.final_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.final_list.count())
        ]
        return initials, finals, self.initial_merge_map, self.final_merge_map

    def _prepare_merge(self, widget: QListWidget, kind: str):
        selected = widget.selectedItems()
        if len(selected) != 1:
            QMessageBox.warning(self, "归并", "请先单击选中一个音标再点击归并。")
            return
        source_symbol = selected[0].data(Qt.ItemDataRole.UserRole)
        self.pending_merge = (kind, source_symbol)
        self.merge_tip.setText(f"已选择“{source_symbol}”，请点击要归并到的目标音标。")

    def _try_finish_merge(self, item: QListWidgetItem, kind: str):
        if self.pending_merge is None:
            return
        pending_kind, source_symbol = self.pending_merge
        if pending_kind != kind:
            return
        target_symbol = item.data(Qt.ItemDataRole.UserRole)
        if not source_symbol or not target_symbol or source_symbol == target_symbol:
            return
        target_list = self.initial_list if kind == "initial" else self.final_list
        for i in range(target_list.count()):
            it = target_list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == source_symbol:
                target_list.takeItem(i)
                break
        if kind == "initial":
            self.initial_merge_map[source_symbol] = target_symbol
        else:
            self.final_merge_map[source_symbol] = target_symbol
        self.pending_merge = None
        self.merge_tip.setText("归并完成。可继续归并或拖拽排序。")


class PhonologyInductionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音系归纳")
        self.resize(332, 206)
        self.service = PhonologyInductionService()
        self.is_dark = True
        self.current_rows = []
        self.current_analysis = None
        self.consonant_only_as_zero_initial = True
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(6)

        title = QLabel("音系归纳")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        root.addWidget(title)

        self.btn_upload = QPushButton("上传调查字表")
        self.btn_upload.setMinimumHeight(38)
        self.btn_generate = QPushButton("生成同音字表")
        self.btn_generate.setMinimumHeight(38)
        self.btn_help = QPushButton("帮助")
        self.btn_help.setMinimumHeight(30)
        self.btn_help.setStyleSheet(
            "QPushButton {background-color:#28a745;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover {background-color:#34c759;}"
            "QPushButton:pressed {background-color:#1f8f3a;}"
        )
        self.btn_help.setEnabled(True)

        root.addWidget(self.btn_upload)
        root.addWidget(self.btn_generate)
        help_row = QHBoxLayout()
        help_row.addStretch()
        help_row.addWidget(self.btn_help)
        root.addLayout(help_row)
        root.addStretch()

        self.btn_upload.clicked.connect(self._on_upload)
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_help.clicked.connect(self._open_help)

    def set_theme(self, is_dark: bool):
        self.is_dark = is_dark

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择调查字表",
            "",
            "表格或文本 (*.xlsx *.xls *.csv *.txt *.tsv);;All Files (*)",
        )
        if not path:
            return
        try:
            header_msg = QMessageBox(self)
            header_msg.setWindowTitle("表头处理")
            header_msg.setIcon(QMessageBox.Icon.Question)
            header_msg.setText("是否跳过第一行（表头）？")
            btn_yes = header_msg.addButton("是", QMessageBox.ButtonRole.YesRole)
            btn_no = header_msg.addButton("否", QMessageBox.ButtonRole.NoRole)
            btn_cancel = header_msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            header_msg.setDefaultButton(btn_yes)
            header_msg.exec()
            clicked = header_msg.clickedButton()
            if clicked is btn_cancel:
                return
            header_choice_yes = clicked is btn_yes
            skip_first_row = header_choice_yes
            rows = self.service.load_rows(path, skip_first_row=skip_first_row)
            if not rows:
                self._show_message("warning", "音系归纳", "未读取到有效数据，请检查文件格式。")
                return
            single_consonant_rows = self.service.find_single_consonant_rows(rows)
            consonant_only_as_zero_initial = True
            if single_consonant_rows:
                consonant_only_as_zero_initial = self._ask_single_consonant_policy(
                    single_consonant_rows
                )
            analysis = self.service.analyze(
                rows,
                consonant_only_as_zero_initial=consonant_only_as_zero_initial,
            )
        except Exception as exc:
            self._show_message("error", "音系归纳", f"读取失败：{exc}")
            return

        self.current_rows = rows
        self.current_analysis = analysis
        self.consonant_only_as_zero_initial = consonant_only_as_zero_initial
        self._show_message(
            "info",
            "音系归纳",
            f"读取成功。\n字条数：{len(rows)}\n不重复音标数：{len(analysis.unique_ipa)}",
        )

    def _on_generate(self):
        if self.current_analysis is None:
            self._show_message("warning", "音系归纳", "请先上传调查字表。")
            return
        tone_dialog = ToneClassDialog(self.current_analysis.unique_tones, self)
        if tone_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        tone_map = tone_dialog.collect_mapping()
        tone_order = tone_dialog.collect_tone_order()

        order_dialog = SymbolOrderDialog(
            initials=self.current_analysis.unique_initials,
            finals=self._default_final_order(self.current_analysis.unique_finals),
            parent=self,
        )
        if order_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        initial_order, final_order, initial_merge_map, final_merge_map = (
            order_dialog.collect_orders()
        )
        analysis_for_export = self.service.apply_symbol_aliases(
            self.current_analysis,
            initial_merge_map=initial_merge_map,
            final_merge_map=final_merge_map,
        )
        initial_order = [x for x in initial_order if x in analysis_for_export.unique_initials]
        final_order = [x for x in final_order if x in analysis_for_export.unique_finals]

        out_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if not out_dir:
            return
        try:
            result = self.service.export_outputs(
                analysis=analysis_for_export,
                tone_class_map=tone_map,
                tone_value_order=tone_order,
                initial_order=initial_order,
                final_order=final_order,
                output_dir=out_dir,
            )
        except Exception as exc:
            self._show_message("error", "音系归纳", f"生成失败：{exc}")
            return

        self._show_message(
            "info",
            "音系归纳",
            "已生成三份结果：\n"
            f"{result.forward_docx_path}\n"
            f"{result.reverse_docx_path}\n"
            f"{result.matrix_xlsx_path}",
        )

    def _ask_single_consonant_policy(self, rows):
        sample = "、".join(f"{row.character}({row.ipa})" for row in rows[:8])
        msg = QMessageBox(self)
        msg.setWindowTitle("单辅音条目处理方式")
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setText(
            "检测到只有单辅音的条目。\n"
            f"示例：{sample}\n\n"
            "请选择处理方式："
        )
        btn_zero = msg.addButton("按零声母字处理", QMessageBox.ButtonRole.YesRole)
        btn_empty = msg.addButton("按空韵字处理", QMessageBox.ButtonRole.NoRole)
        msg.setDefaultButton(btn_zero)
        msg.exec()
        clicked = msg.clickedButton()
        return clicked is btn_zero

    def _open_help(self):
        help_path = get_resource_path(r"Phonetic_Export\index.html")
        if not os.path.exists(help_path):
            self._show_message("warning", "帮助", f"找不到帮助文件:\n{help_path}")
            return
        url = QUrl.fromLocalFile(help_path)
        url.setFragment("s1774519534312")
        opened = QDesktopServices.openUrl(url)
        if not opened:
            self._show_message("warning", "帮助", "帮助页面打开失败。")

    def _default_final_order(self, finals: list[str]) -> list[str]:
        non_empty = [f for f in finals if f]
        empty = [f for f in finals if not f]
        grouped = sorted(non_empty, key=lambda x: (x[0], len(x), x))
        return grouped + empty

    def _show_message(self, level: str, title: str, text: str):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        if level == "error":
            msg.setIcon(QMessageBox.Icon.Critical)
        elif level == "warning":
            msg.setIcon(QMessageBox.Icon.Warning)
        else:
            msg.setIcon(QMessageBox.Icon.Information)
        btn_ok = msg.addButton("确定", QMessageBox.ButtonRole.AcceptRole)
        msg.setDefaultButton(btn_ok)
        msg.exec()
