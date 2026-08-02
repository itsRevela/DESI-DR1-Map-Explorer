# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- Dead "Sky Close-up", "Wide Field", and "SIMBAD" links in the selection info
  panel. `www.legacysurvey.org` now returns 503 for all pages — the sky viewer
  moved to `viewer.legacysurvey.org` (same `ra`/`dec`/`layer`/`zoom` query
  parameters). The SIMBAD link now uses CDS's canonical host
  `simbad.cds.unistra.fr` instead of the deprecated `simbad.u-strasbg.fr`
  alias, whose pages load some assets over plain HTTP and get blocked as
  mixed content in browsers.
- Black (empty) sky map when opening "Sky Close-up" / "Wide Field": the
  relocated viewer no longer offers the `ls-dr10` layer at all, and an unknown
  layer name renders as a black map. Switched to `ls-dr9`, whose tiles are
  served from working S3 buckets for both hemispheres — unlike the new
  default `ls-dr11`, whose full-resolution tiles come from
  `dev-*.viewer.legacysurvey.org` hosts currently serving an invalid
  (Kubernetes ingress placeholder) TLS certificate that browsers reject.
  DR9 is also the Legacy Surveys release DESI DR1 targeting was based on.
- Favorites "Go To" silently failing for real targets: the panel's
  `goto_requested` signal was declared `pyqtSignal(int, ...)`, which is a
  32-bit C++ int — PyQt truncated 64-bit DESI TARGETIDs on emit, so the
  viewer searched the catalog for a mangled ID and found nothing. The signal
  now carries the TARGETID as a Python object, and regression tests cover
  the signal path with a realistic 64-bit TARGETID.
- SIMBAD link almost always answering "No astronomical object found": most
  DESI targets are too faint to be individually catalogued in SIMBAD, so the
  previous 10-arcsec search radius usually matched nothing. Widened to
  2 arcmin (SIMBAD's own form default), which returns the nearest catalogued
  objects sorted by angular distance — the selected object appears at the top
  when it is known, and useful context appears otherwise.
- `setup_and_run.bat` failing on second launch with "the system cannot find the
  batch label specified" for users who downloaded via GitHub ZIP or cloned with
  `core.autocrlf=false`. The repo now ships a `.gitattributes` enforcing
  `*.bat text eol=crlf`, so checkouts and ZIP archives always deliver CRLF
  line endings (Windows CMD's `goto` parser is unreliable on LF-only files).
- Dataset selection branch in `setup_and_run.bat`: `if ... & goto :launch`
  was parsed with `goto :launch` outside the `if`, so picking option 2 fell
  through with `DATASET` empty and the default-to-EDR fallback was unreachable.
  Replaced with parenthesized `if` blocks.

### Added
- Favorites system for bookmarking targets and returning to them later:
  - `B` toggles a favorites panel; `M` (or the "Add to Favorites" link in the
    info panel) favorites/unfavorites the selected target
  - Each favorite can be given a nickname and free-form notes; both are shown
    in the panel and the nickname also appears in the selection info panel
  - Live search box matching nickname, notes, and TARGETID (all query words
    must match), plus an object-type filter (Galaxies / QSOs / Other)
  - Double-click (or "Go To") teleports the camera to ~50 Mpc from the target
    and selects it, switching to the fly camera if needed
  - Stored in `data/favorites.json` (atomic writes; a corrupt file is moved
    aside, never overwritten). New modules `favorites.py` (store, fully unit
    tested) and `favorites_ui.py` (Qt panel)
- Right-click point selection: right-click selects nearest galaxy/QSO to cursor,
  right-click again deselects (toggle behavior)
- Camera auto-rotates to center selected point on screen
- Info panel (top-right) shows TARGETID, redshift, lookback time, comoving/luminosity
  distance, absolute magnitude, solar luminosities, RA/Dec for selected point
- Bright yellow highlight marker on selected point in 3D scene
- KD-tree spatial index (built in background on startup) for fast raycasted picking
  at any dataset size
- TARGETID, TARGET_RA, TARGET_DEC persisted in npz cache (v3) for display on selection

### Changed
- Cache format bumped to points_v3.npz (old v2 cache will be regenerated on next run)
