# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
