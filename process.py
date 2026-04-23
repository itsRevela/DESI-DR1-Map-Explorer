"""Load a DESI zcatalog FITS, filter for reliable cosmological objects, compute
comoving Cartesian positions and absolute r-band magnitudes, cache to .npz.

Cache file is versioned (points_v2.npz) so upgrades invalidate old caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits


CACHE_VERSION = 3
M_SUN_R = 4.65

FLUX_CANDIDATES = (
    "FLUX_R", "FIBERFLUX_R", "FIBERTOTFLUX_R", "DECAM_FLUX_R", "FLUX_G",
)

SPECTYPE_GALAXY = np.uint8(0)
SPECTYPE_QSO = np.uint8(1)
SPECTYPE_OTHER = np.uint8(2)


@dataclass(frozen=True)
class PointCloud:
    """Filtered, transformed DESI catalog ready for rendering."""
    xyz: np.ndarray          # (N, 3) float32, Mpc (comoving)
    spectype: np.ndarray     # (N,) uint8
    z: np.ndarray            # (N,) float32
    M_r: np.ndarray          # (N,) float32, absolute AB magnitude, r-band
    log_L: np.ndarray        # (N,) float32, log10(L / L_sun)
    flux_r: np.ndarray       # (N,) float32, nanomaggies (for flux-mode sizing)
    flux_column: str         # which FITS column was used for flux
    target_id: np.ndarray    # (N,) int64
    target_ra: np.ndarray    # (N,) float32, degrees
    target_dec: np.ndarray   # (N,) float32, degrees
    desi_target: np.ndarray  # (N,) int64, targeting bitmask
    flux_g: np.ndarray       # (N,) float32, g-band nanomaggies
    lookback_gyr: np.ndarray # (N,) float32, lookback time in Gyr

    @property
    def n(self) -> int:
        return int(self.z.shape[0])


def _pick_targeting_column(column_names) -> str | None:
    upper = [c.upper() for c in column_names]
    for candidate in ("DESI_TARGET", "SV3_DESI_TARGET", "SV1_DESI_TARGET"):
        if candidate in upper:
            return column_names[upper.index(candidate)]
    return None


def _pick_flux_column(column_names) -> str | None:
    upper = [c.upper() for c in column_names]
    for candidate in FLUX_CANDIDATES:
        if candidate in upper:
            return column_names[upper.index(candidate)]
    return None


def _normalize_spectype(arr: np.ndarray) -> np.ndarray:
    """Return stripped ascii-bytes for comparison regardless of source dtype."""
    if arr.dtype.kind == "S":
        return np.char.strip(arr)
    if arr.dtype.kind == "U":
        return np.char.encode(np.char.strip(arr), "ascii")
    return np.char.strip(np.char.encode(arr.astype("U"), "ascii"))


def _encode_spectype(spectype_bytes: np.ndarray) -> np.ndarray:
    """Map stripped bytes to uint8 codes."""
    codes = np.full(len(spectype_bytes), SPECTYPE_OTHER, dtype=np.uint8)
    codes[spectype_bytes == b"GALAXY"] = SPECTYPE_GALAXY
    codes[spectype_bytes == b"QSO"] = SPECTYPE_QSO
    return codes


def _spherical_to_cartesian(ra_deg: np.ndarray, dec_deg: np.ndarray,
                            d_mpc: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cos_d = np.cos(dec)
    xyz = np.empty((ra.size, 3), dtype=np.float32)
    xyz[:, 0] = (d_mpc * cos_d * np.cos(ra)).astype(np.float32)
    xyz[:, 1] = (d_mpc * cos_d * np.sin(ra)).astype(np.float32)
    xyz[:, 2] = (d_mpc * np.sin(dec)).astype(np.float32)
    return xyz


def _cache_fresh(fits_path: Path, npz_path: Path) -> bool:
    if not (npz_path.exists() and fits_path.exists()):
        return False
    return npz_path.stat().st_mtime > fits_path.stat().st_mtime


def _load_npz(npz_path: Path) -> PointCloud:
    with np.load(npz_path, allow_pickle=False) as npz:
        flux_column = str(npz["flux_column_name"].item())
        return PointCloud(
            xyz=npz["xyz"].astype(np.float32, copy=False),
            spectype=npz["spectype"].astype(np.uint8, copy=False),
            z=npz["z"].astype(np.float32, copy=False),
            M_r=npz["M_r"].astype(np.float32, copy=False),
            log_L=npz["log_L"].astype(np.float32, copy=False),
            flux_r=npz["flux_r"].astype(np.float32, copy=False),
            flux_column=flux_column,
            target_id=npz["target_id"].astype(np.int64, copy=False),
            target_ra=npz["target_ra"].astype(np.float32, copy=False),
            target_dec=npz["target_dec"].astype(np.float32, copy=False),
            desi_target=npz["desi_target"].astype(np.int64, copy=False),
            flux_g=npz["flux_g"].astype(np.float32, copy=False),
            lookback_gyr=npz["lookback_gyr"].astype(np.float32, copy=False),
        )


def _save_npz(pc: PointCloud, npz_path: Path) -> None:
    # Write to an open file handle so np.savez doesn't append .npz to the tmp name.
    tmp = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            xyz=pc.xyz, spectype=pc.spectype, z=pc.z,
            M_r=pc.M_r, log_L=pc.log_L, flux_r=pc.flux_r,
            flux_column_name=np.array(pc.flux_column),
            target_id=pc.target_id,
            target_ra=pc.target_ra,
            target_dec=pc.target_dec,
            desi_target=pc.desi_target,
            flux_g=pc.flux_g,
            lookback_gyr=pc.lookback_gyr,
        )
    tmp.replace(npz_path)
    print(f"[process] wrote cache: {npz_path} "
          f"({npz_path.stat().st_size / 1e6:.1f} MB)")


def _validate_fits_opens(fits_path: Path) -> None:
    """Quick sanity check that the downloaded file is a parseable FITS."""
    try:
        with fits.open(fits_path, memmap=True) as hdul:
            if len(hdul) < 2:
                raise RuntimeError(f"FITS has no table HDU: {fits_path}")
            _ = hdul[1].columns.names
    except Exception as e:
        raise RuntimeError(
            f"Downloaded file does not open as FITS: {fits_path} ({e!s})"
        ) from e


def _print_summary(pc: PointCloud) -> None:
    n = pc.n
    n_gal = int(np.count_nonzero(pc.spectype == SPECTYPE_GALAXY))
    n_qso = int(np.count_nonzero(pc.spectype == SPECTYPE_QSO))
    n_other = n - n_gal - n_qso

    print("=== Catalog Summary ===")
    print(f"Total         : {n:,}")
    print(f"GALAXY        : {n_gal:,}")
    print(f"QSO           : {n_qso:,}")
    print(f"Other         : {n_other:,}")

    def stat(name, arr, unit=""):
        lo = float(np.min(arr)); med = float(np.median(arr)); hi = float(np.max(arr))
        print(f"{name:13s} min / median / max : "
              f"{lo:10.3f} / {med:10.3f} / {hi:10.3f} {unit}")

    d_comov = np.linalg.norm(pc.xyz, axis=1)
    d_lum = (1.0 + pc.z) * d_comov
    stat("Z", pc.z)
    stat("d_comov", d_comov, "Mpc")
    stat("d_L", d_lum, "Mpc")
    stat("M_r", pc.M_r, "(AB)")
    stat("log_L", pc.log_L, "(L/Lsun)")
    print("=======================")


def _validate_physics(pc: PointCloud) -> None:
    """Hard-stop if magnitudes land in an implausible range."""
    median = float(np.median(pc.M_r))
    mx = float(np.max(pc.M_r))
    errors = []
    if not (-25.0 <= median <= -15.0):
        errors.append(
            f"M_r median = {median:.2f} is outside expected [-25, -15]")
    if mx > 0.0:
        errors.append(
            f"M_r max = {mx:.2f} > 0 (faintest object brighter than the Sun at 10 pc)")
    if errors:
        raise RuntimeError(
            "Physics sanity check failed:\n  " + "\n  ".join(errors)
            + "\n\nCheck unit conversions before rendering."
        )


def _build(fits_path: Path) -> PointCloud:
    _validate_fits_opens(fits_path)

    with fits.open(fits_path, memmap=True) as hdul:
        hdu = hdul[1]
        col_names = list(hdu.columns.names)

        flux_column = _pick_flux_column(col_names)
        if flux_column is None:
            raise RuntimeError(
                f"No usable flux column in {fits_path}. "
                f"Tried {FLUX_CANDIDATES}. Available (first 30): {col_names[:30]}"
            )
        print(f"[process] using flux column: {flux_column}")

        zwarn = np.asarray(hdu.data["ZWARN"])
        zcat_primary = np.asarray(hdu.data["ZCAT_PRIMARY"]).astype(bool, copy=False)
        spectype_raw = np.asarray(hdu.data["SPECTYPE"])
        z_raw = np.asarray(hdu.data["Z"], dtype=np.float64)
        ra_raw = np.asarray(hdu.data["TARGET_RA"], dtype=np.float64)
        dec_raw = np.asarray(hdu.data["TARGET_DEC"], dtype=np.float64)
        flux_raw = np.asarray(hdu.data[flux_column], dtype=np.float64)
        target_id_raw = np.asarray(hdu.data["TARGETID"], dtype=np.int64)
        upper_cols = [c.upper() for c in col_names]
        desi_target_raw = np.zeros(z_raw.size, dtype=np.int64)
        for tgt_col in ("DESI_TARGET", "SV3_DESI_TARGET", "SV1_DESI_TARGET"):
            if tgt_col in upper_cols:
                desi_target_raw |= np.asarray(
                    hdu.data[col_names[upper_cols.index(tgt_col)]],
                    dtype=np.int64)
        n_tgt = int(np.count_nonzero(desi_target_raw))
        print(f"[process] targeting bitmask: {n_tgt:,} / {z_raw.size:,} non-zero")
        if "FLUX_G" in upper_cols:
            flux_g_raw = np.asarray(
                hdu.data[col_names[upper_cols.index("FLUX_G")]],
                dtype=np.float64)
        else:
            flux_g_raw = np.full(z_raw.size, np.nan, dtype=np.float64)

    spectype_norm = _normalize_spectype(spectype_raw)

    mask = (
        (zwarn == 0)
        & zcat_primary
        & (spectype_norm != b"STAR")
        & (z_raw > 0.001)
        & (z_raw < 4.0)
        & (flux_raw > 0.0)
        & np.isfinite(flux_raw)
    )

    n_in = int(z_raw.size)
    n_kept = int(mask.sum())
    print(f"[process] filter: {n_kept:,} / {n_in:,} rows kept "
          f"({n_kept / max(n_in, 1) * 100:.1f}%)")

    z = z_raw[mask].astype(np.float64)
    ra = ra_raw[mask]
    dec = dec_raw[mask]
    flux = flux_raw[mask]
    spec_kept = spectype_norm[mask]
    target_id = target_id_raw[mask]
    desi_target = desi_target_raw[mask]
    flux_g = flux_g_raw[mask].astype(np.float32)

    del zwarn, zcat_primary, spectype_raw, spectype_norm
    del z_raw, ra_raw, dec_raw, flux_raw, target_id_raw
    del desi_target_raw, flux_g_raw

    print(f"[process] computing comoving distances for {n_kept:,} rows "
          f"(Planck18)...")
    d_comov_mpc = Planck18.comoving_distance(z).to_value("Mpc")

    print(f"[process] computing luminosity distances...")
    d_lum_mpc = Planck18.luminosity_distance(z).to_value("Mpc")

    xyz = _spherical_to_cartesian(ra, dec, d_comov_mpc)
    target_ra = ra.astype(np.float32)
    target_dec = dec.astype(np.float32)
    del d_comov_mpc, ra, dec

    # K-corrections ignored — would require per-object SED fitting (out of scope).
    m_r = 22.5 - 2.5 * np.log10(flux)
    d_lum_pc = d_lum_mpc * 1e6
    M_r = (m_r - 5.0 * np.log10(d_lum_pc / 10.0)).astype(np.float32)
    log_L = (0.4 * (M_SUN_R - M_r)).astype(np.float32)
    del d_lum_mpc, d_lum_pc, m_r

    print(f"[process] computing lookback times...")
    lookback_gyr = Planck18.lookback_time(z).to_value("Gyr").astype(np.float32)

    pc = PointCloud(
        xyz=xyz,
        spectype=_encode_spectype(spec_kept),
        z=z.astype(np.float32),
        M_r=M_r,
        log_L=log_L,
        flux_r=flux.astype(np.float32),
        flux_column=flux_column,
        target_id=target_id,
        target_ra=target_ra,
        target_dec=target_dec,
        desi_target=desi_target,
        flux_g=flux_g,
        lookback_gyr=lookback_gyr,
    )

    _print_summary(pc)
    _validate_physics(pc)
    return pc


def load_or_build(fits_path: Path, npz_path: Path) -> PointCloud:
    fits_path = Path(fits_path)
    npz_path = Path(npz_path)

    if _cache_fresh(fits_path, npz_path):
        print(f"[process] cache hit: {npz_path}")
        return _load_npz(npz_path)

    print(f"[process] building {npz_path} from {fits_path}")
    pc = _build(fits_path)
    _save_npz(pc, npz_path)
    return pc


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Process DESI FITS to npz cache.")
    p.add_argument("--fits", default="data/zall-pix-fuji.fits")
    p.add_argument("--npz", default="data/points_v2.npz")
    args = p.parse_args()
    pc = load_or_build(Path(args.fits), Path(args.npz))
    print(f"[process] done: {pc.n:,} points, xyz.shape={pc.xyz.shape}, "
          f"dtype={pc.xyz.dtype}, flux_column={pc.flux_column}")
