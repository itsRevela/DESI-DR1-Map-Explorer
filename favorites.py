"""Persistent favorites (bookmarks) for DESI targets.

Pure-logic module (no Qt): a FavoritesStore holds immutable FavoriteEntry
records keyed by (dataset, target_id) and persists them to a JSON file.
Every mutation saves immediately via an atomic write, so a crash never
loses more than the in-flight change.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

FORMAT_VERSION = 1


@dataclass(frozen=True)
class FavoriteEntry:
    """One bookmarked target. Display fields (spectype, z, ra, dec) are
    cached at bookmark time so the list renders without catalog lookups."""

    target_id: int
    dataset: str
    spectype: str = "?"
    z: float = 0.0
    ra: float = 0.0
    dec: float = 0.0
    nickname: str = ""
    notes: str = ""
    added_at: str = ""

    @property
    def key(self) -> tuple[str, int]:
        return (self.dataset, self.target_id)

    @property
    def display_name(self) -> str:
        return self.nickname if self.nickname else f"TID {self.target_id}"

    def matches(self, query: str) -> bool:
        """Case-insensitive AND-match of all query tokens against
        nickname, notes, and TARGETID."""
        tokens = query.strip().lower().split()
        if not tokens:
            return True
        haystack = f"{self.nickname}\n{self.notes}\n{self.target_id}".lower()
        return all(tok in haystack for tok in tokens)


def _entry_from_dict(raw: dict) -> FavoriteEntry:
    return FavoriteEntry(
        target_id=int(raw["target_id"]),
        dataset=str(raw["dataset"]),
        spectype=str(raw.get("spectype", "?")),
        z=float(raw.get("z", 0.0)),
        ra=float(raw.get("ra", 0.0)),
        dec=float(raw.get("dec", 0.0)),
        nickname=str(raw.get("nickname", "")),
        notes=str(raw.get("notes", "")),
        added_at=str(raw.get("added_at", "")),
    )


class FavoritesStore:
    """Repository for FavoriteEntry records with JSON persistence."""

    def __init__(self, path: Path,
                 entries: dict[tuple[str, int], FavoriteEntry] | None = None):
        self.path = Path(path)
        self._entries: dict[tuple[str, int], FavoriteEntry] = dict(entries or {})

    # -- persistence ----------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> "FavoritesStore":
        """Load a store from *path*. A missing file yields an empty store.
        A corrupt file is moved aside (never overwritten) and the store
        starts empty."""
        path = Path(path)
        if not path.exists():
            return cls(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = {}
            for raw in data["favorites"]:
                entry = _entry_from_dict(raw)
                entries[entry.key] = entry
            return cls(path, entries)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            backup = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            path.rename(backup)
            print(f"[favorites] WARNING: could not parse {path} ({exc}); "
                  f"moved it to {backup} and starting with an empty list")
            return cls(path)

    def save(self) -> None:
        """Atomic write: serialize to a temp file, then replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FORMAT_VERSION,
            "favorites": [dataclasses.asdict(e) for e in self.all()],
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # -- queries ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, dataset: str, target_id: int) -> FavoriteEntry | None:
        return self._entries.get((dataset, int(target_id)))

    def is_favorite(self, dataset: str, target_id: int) -> bool:
        return (dataset, int(target_id)) in self._entries

    def all(self) -> list[FavoriteEntry]:
        """All entries, newest first (ISO timestamps sort lexicographically)."""
        return sorted(self._entries.values(),
                      key=lambda e: (e.added_at, e.target_id), reverse=True)

    def search(self, query: str = "",
               spectype: str | None = None) -> list[FavoriteEntry]:
        """Entries matching *query* (nickname/notes/TARGETID) and, if given,
        an exact *spectype* ("GALAXY" | "QSO" | "OTHER"). Newest first."""
        result = [e for e in self.all() if e.matches(query)]
        if spectype:
            result = [e for e in result if e.spectype == spectype]
        return result

    # -- mutations ---------------------------------------------------------------

    def add(self, entry: FavoriteEntry) -> FavoriteEntry:
        if not entry.added_at:
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            entry = dataclasses.replace(entry, added_at=stamp)
        self._entries[entry.key] = entry
        self.save()
        return entry

    def remove(self, dataset: str, target_id: int) -> bool:
        if self._entries.pop((dataset, int(target_id)), None) is None:
            return False
        self.save()
        return True

    def toggle(self, entry: FavoriteEntry) -> bool:
        """Add *entry* if absent, remove it if present.
        Returns True when the target is now a favorite."""
        if self.is_favorite(entry.dataset, entry.target_id):
            self.remove(entry.dataset, entry.target_id)
            return False
        self.add(entry)
        return True

    def update(self, dataset: str, target_id: int, *,
               nickname: str | None = None,
               notes: str | None = None) -> FavoriteEntry | None:
        """Replace nickname and/or notes on an existing entry.
        Returns the new entry, or None if the target is not favorited."""
        current = self.get(dataset, target_id)
        if current is None:
            return None
        changes: dict[str, str] = {}
        if nickname is not None:
            changes["nickname"] = nickname
        if notes is not None:
            changes["notes"] = notes
        updated = dataclasses.replace(current, **changes)
        self._entries[updated.key] = updated
        self.save()
        return updated
