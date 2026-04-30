# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
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
