"""Tests for favorites.py: persistence, toggle, search/filter, editing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from favorites import FavoriteEntry, FavoritesStore  # noqa: E402


def _entry(tid: int = 1, dataset: str = "dr1", **kw) -> FavoriteEntry:
    defaults = dict(
        target_id=tid, dataset=dataset, spectype="GALAXY",
        z=0.42, ra=161.066208, dec=-1.552792,
        nickname="", notes="", added_at="2026-08-01T12:00:00",
    )
    defaults.update(kw)
    return FavoriteEntry(**defaults)


# -- toggle / add / remove ----------------------------------------------------

def test_toggle_adds_then_removes(tmp_path: Path) -> None:
    store = FavoritesStore.load(tmp_path / "fav.json")
    entry = _entry()

    assert store.toggle(entry) is True
    assert store.is_favorite("dr1", 1)

    assert store.toggle(entry) is False
    assert not store.is_favorite("dr1", 1)


def test_add_stamps_added_at_when_missing(tmp_path: Path) -> None:
    store = FavoritesStore.load(tmp_path / "fav.json")
    store.add(_entry(added_at=""))
    saved = store.get("dr1", 1)
    assert saved is not None and saved.added_at != ""


def test_remove_missing_entry_is_noop(tmp_path: Path) -> None:
    store = FavoritesStore.load(tmp_path / "fav.json")
    assert store.remove("dr1", 999) is False


def test_same_target_id_distinct_per_dataset(tmp_path: Path) -> None:
    store = FavoritesStore.load(tmp_path / "fav.json")
    store.add(_entry(tid=7, dataset="dr1"))
    store.add(_entry(tid=7, dataset="edr"))
    assert len(store) == 2
    assert store.remove("edr", 7) is True
    assert store.is_favorite("dr1", 7)


# -- persistence --------------------------------------------------------------

def test_roundtrip_persistence(tmp_path: Path) -> None:
    path = tmp_path / "fav.json"
    store = FavoritesStore.load(path)
    store.add(_entry(nickname="Sombrero twin", notes="weird spiral arms"))

    reloaded = FavoritesStore.load(path)
    entry = reloaded.get("dr1", 1)
    assert entry is not None
    assert entry.nickname == "Sombrero twin"
    assert entry.notes == "weird spiral arms"
    assert entry.z == pytest.approx(0.42)


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    store = FavoritesStore.load(tmp_path / "does-not-exist.json")
    assert len(store) == 0


def test_load_corrupt_file_preserves_backup(tmp_path: Path) -> None:
    path = tmp_path / "fav.json"
    path.write_text("{not valid json", encoding="utf-8")

    store = FavoritesStore.load(path)
    assert len(store) == 0
    backups = list(tmp_path.glob("fav.json.corrupt*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{not valid json"


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "fav.json"
    store = FavoritesStore.load(path)
    store.add(_entry())
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["favorites"][0]["target_id"] == 1


# -- search / filter ----------------------------------------------------------

def _seeded_store(tmp_path: Path) -> FavoritesStore:
    store = FavoritesStore.load(tmp_path / "fav.json")
    store.add(_entry(tid=101, spectype="GALAXY", nickname="Red Ellipse",
                     notes="candidate lens", added_at="2026-08-01T10:00:00"))
    store.add(_entry(tid=202, spectype="QSO", nickname="",
                     notes="very high redshift", added_at="2026-08-01T11:00:00"))
    store.add(_entry(tid=303, spectype="OTHER", nickname="mystery blob",
                     notes="", added_at="2026-08-01T12:00:00"))
    return store


def test_search_empty_query_returns_all_newest_first(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    result = store.search("")
    assert [e.target_id for e in result] == [303, 202, 101]


def test_search_matches_nickname_case_insensitive(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    result = store.search("RED ellipse")
    assert [e.target_id for e in result] == [101]


def test_search_matches_notes(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    result = store.search("redshift")
    assert [e.target_id for e in result] == [202]


def test_search_matches_target_id(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    result = store.search("303")
    assert [e.target_id for e in result] == [303]


def test_search_multi_token_requires_all_tokens(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    assert [e.target_id for e in store.search("candidate lens")] == [101]
    assert store.search("candidate blob") == []


def test_filter_by_spectype(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    result = store.search("", spectype="QSO")
    assert [e.target_id for e in result] == [202]


def test_search_and_filter_combined(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    assert store.search("blob", spectype="QSO") == []
    assert [e.target_id for e in store.search("blob", spectype="OTHER")] == [303]


# -- editing ------------------------------------------------------------------

def test_update_nickname_and_notes_persists(tmp_path: Path) -> None:
    path = tmp_path / "fav.json"
    store = FavoritesStore.load(path)
    store.add(_entry())

    updated = store.update("dr1", 1, nickname="Neo", notes="follow up")
    assert updated is not None
    assert updated.nickname == "Neo"

    reloaded = FavoritesStore.load(path)
    entry = reloaded.get("dr1", 1)
    assert entry is not None
    assert entry.nickname == "Neo"
    assert entry.notes == "follow up"


def test_update_preserves_unspecified_fields(tmp_path: Path) -> None:
    store = FavoritesStore.load(tmp_path / "fav.json")
    store.add(_entry(nickname="keep me", notes="original"))
    updated = store.update("dr1", 1, notes="changed")
    assert updated is not None
    assert updated.nickname == "keep me"
    assert updated.notes == "changed"


def test_update_missing_entry_returns_none(tmp_path: Path) -> None:
    store = FavoritesStore.load(tmp_path / "fav.json")
    assert store.update("dr1", 999, nickname="ghost") is None


def test_entries_are_immutable(tmp_path: Path) -> None:
    store = FavoritesStore.load(tmp_path / "fav.json")
    store.add(_entry())
    entry = store.get("dr1", 1)
    with pytest.raises(Exception):
        entry.nickname = "mutated"  # type: ignore[misc]
