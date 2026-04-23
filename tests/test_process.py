"""Smoke tests for process.py: filter mask, luminosity math, cache behavior."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.cosmology import Planck18
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process import (  # noqa: E402
    M_SUN_R, SPECTYPE_GALAXY, SPECTYPE_OTHER, SPECTYPE_QSO,
    _load_npz, _pick_flux_column, _spherical_to_cartesian,
    load_or_build,
)


def _make_fake_fits(path: Path, rows: list[dict]) -> None:
    """Build a minimal DESI-shaped FITS with the given rows."""
    n = len(rows)
    col_defs = [
        fits.Column(name="ZWARN", format="K",
                    array=np.array([r["ZWARN"] for r in rows], dtype=np.int64)),
        fits.Column(name="ZCAT_PRIMARY", format="L",
                    array=np.array([r["ZCAT_PRIMARY"] for r in rows], dtype=bool)),
        fits.Column(name="SPECTYPE", format="6A",
                    array=np.array([r["SPECTYPE"] for r in rows])),
        fits.Column(name="Z", format="D",
                    array=np.array([r["Z"] for r in rows], dtype=np.float64)),
        fits.Column(name="TARGET_RA", format="D",
                    array=np.array([r["TARGET_RA"] for r in rows], dtype=np.float64)),
        fits.Column(name="TARGET_DEC", format="D",
                    array=np.array([r["TARGET_DEC"] for r in rows], dtype=np.float64)),
        fits.Column(name="FLUX_R", format="E",
                    array=np.array([r["FLUX_R"] for r in rows], dtype=np.float32)),
        fits.Column(name="TARGETID", format="K",
                    array=np.array([r.get("TARGETID", i) for i, r in enumerate(rows)],
                                   dtype=np.int64)),
        fits.Column(name="DESI_TARGET", format="K",
                    array=np.array([r.get("DESI_TARGET", 0) for r in rows],
                                   dtype=np.int64)),
        fits.Column(name="FLUX_G", format="E",
                    array=np.array([r.get("FLUX_G", 3.0) for r in rows],
                                   dtype=np.float32)),
    ]
    hdu = fits.BinTableHDU.from_columns(col_defs)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def _good_row(**kw):
    base = dict(
        ZWARN=0, ZCAT_PRIMARY=True, SPECTYPE="GALAXY",
        Z=0.5, TARGET_RA=10.0, TARGET_DEC=20.0, FLUX_R=5.0,
    )
    base.update(kw)
    return base


def test_pick_flux_column_prefers_flux_r():
    assert _pick_flux_column(["FLUX_G", "FLUX_R", "FIBERFLUX_R"]) == "FLUX_R"


def test_pick_flux_column_falls_back_in_order():
    assert _pick_flux_column(["FIBERFLUX_R", "FLUX_G"]) == "FIBERFLUX_R"
    assert _pick_flux_column(["FLUX_G"]) == "FLUX_G"
    assert _pick_flux_column(["UNRELATED"]) is None


def test_filter_drops_stars_zwarn_zcat_and_bad_z(tmp_path):
    rows = [
        _good_row(),                                           # keep
        _good_row(SPECTYPE="STAR"),                            # drop: star
        _good_row(ZWARN=4),                                    # drop: zwarn
        _good_row(ZCAT_PRIMARY=False),                         # drop: duplicate
        _good_row(Z=0.0005),                                   # drop: too-low z
        _good_row(Z=4.5),                                      # drop: too-high z
        _good_row(FLUX_R=-0.1),                                # drop: neg flux
        _good_row(FLUX_R=0.0),                                 # drop: zero flux
        _good_row(SPECTYPE="QSO", Z=2.0, FLUX_R=0.3),          # keep
    ]
    fits_path = tmp_path / "fake.fits"
    npz_path = tmp_path / "points_v2.npz"
    _make_fake_fits(fits_path, rows)

    pc = load_or_build(fits_path, npz_path)

    assert pc.n == 2
    assert set(pc.spectype.tolist()) == {int(SPECTYPE_GALAXY), int(SPECTYPE_QSO)}


def test_xyz_geometry_matches_planck18(tmp_path):
    # Single row at known RA/Dec/Z — verify (x,y,z) = d_comoving * unit vector.
    z = 0.5
    ra_deg = 10.0
    dec_deg = 20.0
    rows = [_good_row(Z=z, TARGET_RA=ra_deg, TARGET_DEC=dec_deg)]
    fits_path = tmp_path / "fake.fits"
    npz_path = tmp_path / "points_v2.npz"
    _make_fake_fits(fits_path, rows)

    pc = load_or_build(fits_path, npz_path)
    assert pc.n == 1

    d_comov_expected = Planck18.comoving_distance(z).to_value("Mpc")
    radius = float(np.linalg.norm(pc.xyz[0]))
    assert abs(radius - d_comov_expected) / d_comov_expected < 1e-4

    # direction
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    expected = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ], dtype=np.float32)
    got = pc.xyz[0] / radius
    assert np.allclose(got, expected, atol=1e-4)


def test_absolute_magnitude_math(tmp_path):
    # m_r = 22.5 - 2.5 log10(FLUX_R); M_r = m_r - 5 log10(d_L_pc / 10)
    flux = 10.0
    z = 0.1
    rows = [_good_row(Z=z, FLUX_R=flux)]
    fits_path = tmp_path / "fake.fits"
    npz_path = tmp_path / "points_v2.npz"
    _make_fake_fits(fits_path, rows)

    pc = load_or_build(fits_path, npz_path)

    d_L_mpc = float(Planck18.luminosity_distance(z).to_value("Mpc"))
    m_r_expected = 22.5 - 2.5 * np.log10(flux)
    M_r_expected = m_r_expected - 5.0 * np.log10(d_L_mpc * 1e6 / 10.0)
    log_L_expected = 0.4 * (M_SUN_R - M_r_expected)

    assert abs(float(pc.M_r[0]) - M_r_expected) < 1e-3
    assert abs(float(pc.log_L[0]) - log_L_expected) < 1e-3


def test_cache_hit_skips_rebuild(tmp_path):
    rows = [_good_row() for _ in range(5)]
    fits_path = tmp_path / "fake.fits"
    npz_path = tmp_path / "points_v2.npz"
    _make_fake_fits(fits_path, rows)

    pc1 = load_or_build(fits_path, npz_path)
    mtime1 = npz_path.stat().st_mtime_ns

    # second call should use cache — no rewrite
    pc2 = load_or_build(fits_path, npz_path)
    mtime2 = npz_path.stat().st_mtime_ns
    assert mtime1 == mtime2
    assert pc1.n == pc2.n
    np.testing.assert_array_equal(pc1.xyz, pc2.xyz)


def test_cache_invalidates_when_fits_is_newer(tmp_path):
    rows = [_good_row() for _ in range(3)]
    fits_path = tmp_path / "fake.fits"
    npz_path = tmp_path / "points_v2.npz"
    _make_fake_fits(fits_path, rows)

    pc1 = load_or_build(fits_path, npz_path)
    npz_mtime = npz_path.stat().st_mtime_ns

    # Touch FITS to make it newer than the cache
    new_mtime = npz_mtime + 10_000_000_000    # +10s in ns
    os.utime(fits_path, ns=(new_mtime, new_mtime))

    pc2 = load_or_build(fits_path, npz_path)
    assert npz_path.stat().st_mtime_ns > npz_mtime
    assert pc2.n == pc1.n


def test_physics_validation_raises_on_absurd_flux(tmp_path):
    # A flux of 1e-12 nMgy gives m_r ~ 52, and at z=0.5 that yields M_r > 0
    # (implausibly faint) — the physics gate should fire.
    rows = [_good_row(Z=0.5, FLUX_R=1e-12)]
    fits_path = tmp_path / "fake.fits"
    npz_path = tmp_path / "points_v2.npz"
    _make_fake_fits(fits_path, rows)

    with pytest.raises(RuntimeError, match="sanity"):
        load_or_build(fits_path, npz_path)


def test_spherical_to_cartesian_pole():
    # RA is irrelevant at the pole (dec=90); x,y should be ~0, z = d.
    xyz = _spherical_to_cartesian(
        ra_deg=np.array([37.0]),
        dec_deg=np.array([90.0]),
        d_mpc=np.array([1000.0]),
    )
    assert abs(xyz[0, 0]) < 1e-3
    assert abs(xyz[0, 1]) < 1e-3
    assert abs(xyz[0, 2] - 1000.0) < 1e-3
