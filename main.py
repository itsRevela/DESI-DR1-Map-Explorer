"""DESI DR1 3D Map Explorer — entry point.

Orchestrates: download (if missing) -> process/cache -> viewer.
Default dataset is dr1 (full ~18M catalog). Use --dataset edr for the
smaller ~1.2M fuji catalog while iterating.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from download import DR1_URL, EDR_URL, ensure_fits
from process import load_or_build
from viewer import run_viewer


DATASETS = {
    "edr": ("zall-pix-fuji.fits", EDR_URL,
            "EDR / fuji (~2 GB, ~1.2M filtered points)"),
    "dr1": ("zall-pix-iron.fits", DR1_URL,
            "DR1 / iron (~21 GB, ~18M filtered points)"),
}


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="DESI 3D map explorer (download + process + view).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Datasets:\n  edr  = " + DATASETS["edr"][2]
               + "\n  dr1  = " + DATASETS["dr1"][2]
               + "\n\nCamera controls: W/A/S/D move, F/C up/down, Shift boost,"
               + " LMB drag look, wheel speed, T cam swap, L size toggle,"
               + " V color cycle, K LOD toggle, H help, Esc quit",
    )
    p.add_argument("--dataset", choices=list(DATASETS.keys()), default="dr1",
                   help="dataset to explore (default: dr1)")
    p.add_argument("--data-dir", default="data",
                   help="directory for FITS + npz cache (default: ./data)")
    p.add_argument("--no-view", action="store_true",
                   help="download + process only, skip viewer (useful for priming cache)")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    fits_name, url, desc = DATASETS[args.dataset]
    fits_path = data_dir / fits_name
    npz_path = data_dir / "points_v4.npz"

    print(f"=== DESI Map Explorer — dataset: {args.dataset} ===")
    print(f"[main] {desc}")
    print(f"[main] FITS: {fits_path}")
    print(f"[main] cache: {npz_path}")
    print(f"[main] startup rss: {_rss_mb():.0f} MB")

    ensure_fits(url, fits_path)

    pc = load_or_build(fits_path, npz_path)

    print(f"[main] post-load rss: {_rss_mb():.0f} MB  "
          f"points: {pc.n:,}  flux_column: {pc.flux_column}")

    if args.no_view:
        print("[main] --no-view set; exiting before viewer.")
        return 0

    return run_viewer(pc)


if __name__ == "__main__":
    sys.exit(main())
