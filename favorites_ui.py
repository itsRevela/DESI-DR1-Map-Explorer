"""Qt overlay panel for browsing, searching, and editing favorite targets.

Presentation only: all data lives in a FavoritesStore (favorites.py).
The viewer connects to `goto_requested` to fly to a target and to
`favorites_changed` to refresh the selection info panel.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from favorites import FavoritesStore

PANEL_WIDTH = 380

_TYPE_FILTERS: list[tuple[str, str | None]] = [
    ("All types", None),
    ("Galaxies", "GALAXY"),
    ("QSOs", "QSO"),
    ("Other", "OTHER"),
]

_PANEL_STYLE = """
QWidget#favPanel {
  background-color: rgba(5, 5, 20, 210);
  border: 1px solid rgba(80, 80, 200, 120);
  border-radius: 6px;
}
QLabel { color: rgba(255, 255, 255, 220); background: transparent; }
QLineEdit, QTextEdit, QComboBox, QListWidget {
  background-color: rgba(15, 15, 45, 220);
  color: #eeeedd;
  border: 1px solid rgba(80, 80, 200, 120);
  border-radius: 4px;
  padding: 3px;
  selection-background-color: rgba(80, 80, 200, 160);
}
QListWidget::item { padding: 3px; }
QListWidget::item:selected { background: rgba(80, 80, 200, 140); }
QComboBox QAbstractItemView {
  background-color: rgb(15, 15, 45);
  color: #eeeedd;
}
QPushButton {
  background-color: rgba(40, 40, 90, 220);
  color: #eeeedd;
  border: 1px solid rgba(80, 80, 200, 120);
  border-radius: 4px;
  padding: 4px 10px;
}
QPushButton:hover { background-color: rgba(60, 60, 130, 220); }
QPushButton#removeBtn { color: #ff9988; }
"""


class FavoritesPanel(QWidget):
    """Searchable, filterable list of favorites with a nickname/notes editor."""

    # target_id must travel as a Python object: DESI TARGETIDs are 64-bit
    # values, and pyqtSignal(int) is a C++ 32-bit int that silently
    # truncates them.
    goto_requested = pyqtSignal(object, str)   # (target_id, dataset)
    favorites_changed = pyqtSignal()

    def __init__(self, store: FavoritesStore, dataset: str,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._store = store
        self._dataset = dataset

        self.setObjectName("favPanel")
        self.setStyleSheet(_PANEL_STYLE)
        self.setFixedWidth(PANEL_WIDTH)
        font = self.font()
        font.setFamily("Consolas")
        font.setPointSize(9)
        self.setFont(font)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        self._title = QLabel("★ Favorites")
        self._title.setStyleSheet("font-weight: bold; color: #ffdd88;")
        layout.addWidget(self._title)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search nickname, notes, TARGETID…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self.refresh)
        layout.addWidget(self._search)

        self._type_filter = QComboBox()
        for label, _ in _TYPE_FILTERS:
            self._type_filter.addItem(label)
        self._type_filter.currentIndexChanged.connect(self.refresh)
        layout.addWidget(self._type_filter)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_item_selected)
        self._list.itemDoubleClicked.connect(self._on_item_activated)
        layout.addWidget(self._list, stretch=1)

        self._nickname = QLineEdit()
        self._nickname.setPlaceholderText("Nickname")
        layout.addWidget(self._nickname)

        self._notes = QTextEdit()
        self._notes.setPlaceholderText("Notes")
        self._notes.setFixedHeight(70)
        self._notes.setTabChangesFocus(True)
        layout.addWidget(self._notes)

        buttons = QHBoxLayout()
        self._goto_btn = QPushButton("Go To")
        self._goto_btn.clicked.connect(self._on_goto_clicked)
        buttons.addWidget(self._goto_btn)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_save_clicked)
        buttons.addWidget(self._save_btn)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setObjectName("removeBtn")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        buttons.addWidget(self._remove_btn)
        layout.addLayout(buttons)

        hint = QLabel("M marks selected target · B toggles this panel\n"
                      "Double-click an entry to fly to it")
        hint.setStyleSheet("color: #8888aa;")
        layout.addWidget(hint)

        self._set_editor_enabled(False)
        self.refresh()

    # -- helpers ---------------------------------------------------------------

    def _current_key(self) -> tuple[str, int] | None:
        item = self._list.currentItem()
        if item is None:
            return None
        dataset, target_id = item.data(Qt.ItemDataRole.UserRole)
        return (dataset, int(target_id))

    def _selected_spectype(self) -> str | None:
        return _TYPE_FILTERS[self._type_filter.currentIndex()][1]

    def _set_editor_enabled(self, on: bool) -> None:
        for w in (self._nickname, self._notes, self._goto_btn,
                  self._save_btn, self._remove_btn):
            w.setEnabled(on)
        if not on:
            self._nickname.clear()
            self._notes.clear()

    # -- refresh ----------------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the list from the store, honoring search + type filter."""
        previous = self._current_key()
        entries = self._store.search(self._search.text(),
                                     spectype=self._selected_spectype())

        self._list.blockSignals(True)
        self._list.clear()
        restored_row = -1
        for row, entry in enumerate(entries):
            label = f"★ {entry.display_name}  ·  " \
                    f"{entry.spectype}  z={entry.z:.3f}"
            if entry.dataset != self._dataset:
                label += f"  [{entry.dataset}]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole,
                         (entry.dataset, entry.target_id))
            tooltip = (f"TARGETID {entry.target_id}\n"
                       f"RA/Dec {entry.ra:.4f} / {entry.dec:.4f}\n"
                       f"added {entry.added_at}")
            if entry.notes:
                tooltip += f"\n\n{entry.notes}"
            item.setToolTip(tooltip)
            self._list.addItem(item)
            if previous == (entry.dataset, entry.target_id):
                restored_row = row
        self._list.blockSignals(False)

        total = len(self._store)
        shown = len(entries)
        suffix = f" ({shown}/{total})" if shown != total else f" ({total})"
        self._title.setText(f"★ Favorites{suffix}")

        if restored_row >= 0:
            self._list.setCurrentRow(restored_row)
        else:
            self._set_editor_enabled(False)

    # -- slots --------------------------------------------------------------------

    def _on_item_selected(self, current: QListWidgetItem | None,
                          _previous: QListWidgetItem | None = None) -> None:
        if current is None:
            self._set_editor_enabled(False)
            return
        dataset, target_id = current.data(Qt.ItemDataRole.UserRole)
        entry = self._store.get(dataset, target_id)
        if entry is None:
            self._set_editor_enabled(False)
            return
        self._set_editor_enabled(True)
        self._nickname.setText(entry.nickname)
        self._notes.setPlainText(entry.notes)

    def _on_item_activated(self, item: QListWidgetItem) -> None:
        dataset, target_id = item.data(Qt.ItemDataRole.UserRole)
        self.goto_requested.emit(int(target_id), dataset)

    def _on_goto_clicked(self) -> None:
        key = self._current_key()
        if key is not None:
            self.goto_requested.emit(key[1], key[0])

    def _on_save_clicked(self) -> None:
        key = self._current_key()
        if key is None:
            return
        self._store.update(key[0], key[1],
                           nickname=self._nickname.text().strip(),
                           notes=self._notes.toPlainText().strip())
        self.refresh()
        self.favorites_changed.emit()

    def _on_remove_clicked(self) -> None:
        key = self._current_key()
        if key is None:
            return
        self._store.remove(key[0], key[1])
        self.refresh()
        self.favorites_changed.emit()
