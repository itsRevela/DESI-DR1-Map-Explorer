"""Tests for favorites_ui.py: signal integrity with real-scale TARGETIDs.

Runs headless via the offscreen Qt platform plugin.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from favorites import FavoriteEntry, FavoritesStore  # noqa: E402
from favorites_ui import FavoritesPanel  # noqa: E402

# Real DESI TARGETIDs encode RELEASE/BRICKID/OBJID in 64 bits — far beyond
# the 32-bit range a naive pyqtSignal(int) can carry.
BIG_TID = 39627745205683897


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _store_with_big_tid(tmp_path: Path) -> FavoritesStore:
    store = FavoritesStore.load(tmp_path / "fav.json")
    store.add(FavoriteEntry(
        target_id=BIG_TID, dataset="dr1", spectype="QSO", z=2.5,
        ra=161.066208, dec=-1.552792, added_at="2026-08-01T12:00:00",
    ))
    return store


def test_goto_button_emits_full_64bit_target_id(tmp_path: Path,
                                                qapp: QApplication) -> None:
    panel = FavoritesPanel(_store_with_big_tid(tmp_path), "dr1")
    received: list[tuple[int, str]] = []
    panel.goto_requested.connect(lambda tid, ds: received.append((tid, ds)))

    panel._list.setCurrentRow(0)
    panel._on_goto_clicked()

    assert received == [(BIG_TID, "dr1")]


def test_double_click_emits_full_64bit_target_id(tmp_path: Path,
                                                 qapp: QApplication) -> None:
    panel = FavoritesPanel(_store_with_big_tid(tmp_path), "dr1")
    received: list[tuple[int, str]] = []
    panel.goto_requested.connect(lambda tid, ds: received.append((tid, ds)))

    panel._on_item_activated(panel._list.item(0))

    assert received == [(BIG_TID, "dr1")]


def test_search_finds_big_target_id(tmp_path: Path,
                                    qapp: QApplication) -> None:
    panel = FavoritesPanel(_store_with_big_tid(tmp_path), "dr1")
    panel._search.setText(str(BIG_TID))
    assert panel._list.count() == 1
