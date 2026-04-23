"""Streaming FITS downloader with HTTP Range-based resume and a tqdm progress bar.

No checksum verification — DESI does not publish inline checksums for these
catalogs. Final file is validated by a full-length check and (later, in
process.py) by opening it with astropy.io.fits.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
from tqdm import tqdm


EDR_URL = "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/zcatalog/zall-pix-fuji.fits"
DR1_URL = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/zall-pix-iron.fits"

CHUNK = 1 << 20           # 1 MiB
CONNECT_TIMEOUT = 15.0    # seconds
READ_TIMEOUT = 120.0      # seconds
MAX_ATTEMPTS = 4


def _head_size(url: str) -> int | None:
    """Return Content-Length from HEAD, or None if the server doesn't expose it."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=CONNECT_TIMEOUT)
        r.raise_for_status()
        if "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except (requests.RequestException, ValueError):
        pass
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0"},
                         stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        if r.status_code in (200, 206):
            cr = r.headers.get("Content-Range")
            if cr and "/" in cr:
                total = cr.rsplit("/", 1)[1]
                if total.isdigit():
                    return int(total)
            if "Content-Length" in r.headers and r.status_code == 200:
                return int(r.headers["Content-Length"])
    except requests.RequestException:
        pass
    return None


def _stream_to(part_path: Path, url: str, total: int | None,
               already: int, desc: str) -> None:
    """Stream bytes to part_path, appending from already, with a tqdm bar."""
    headers = {}
    if already > 0:
        headers["Range"] = f"bytes={already}-"

    with requests.get(url, headers=headers, stream=True,
                      timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
        r.raise_for_status()

        if already > 0 and r.status_code == 200:
            tqdm.write(f"[download] server ignored Range; restarting from 0")
            part_path.unlink(missing_ok=True)
            already = 0
        elif already > 0 and r.status_code != 206:
            raise requests.HTTPError(
                f"expected 206 on resume, got {r.status_code}")

        mode = "ab" if already > 0 else "wb"
        with open(part_path, mode) as f, tqdm(
            total=total,
            initial=already,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=desc, miniters=1,
        ) as bar:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))


def ensure_fits(url: str, dest_path: Path) -> Path:
    """Download url to dest_path if missing; resume from dest_path.part on restart.

    Returns the final path. Raises requests.RequestException / OSError on
    irrecoverable failure.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    total = _head_size(url)

    if dest_path.exists():
        size = dest_path.stat().st_size
        if total is None or size == total:
            print(f"[download] already present: {dest_path} "
                  f"({size / 1e9:.2f} GB)")
            return dest_path
        print(f"[download] size mismatch on {dest_path} "
              f"({size} vs expected {total}); re-downloading")
        dest_path.unlink()

    desc = dest_path.name

    for attempt in range(1, MAX_ATTEMPTS + 1):
        already = part_path.stat().st_size if part_path.exists() else 0
        if total is not None and already > total:
            print(f"[download] .part larger than expected; discarding")
            part_path.unlink()
            already = 0
        if total is not None and already == total:
            break
        try:
            _stream_to(part_path, url, total, already, desc)
            break
        except (requests.RequestException, OSError) as e:
            if attempt == MAX_ATTEMPTS:
                raise
            backoff = 2 ** attempt
            print(f"[download] attempt {attempt} failed ({e!s}); "
                  f"retrying in {backoff}s")
            time.sleep(backoff)

    final_size = part_path.stat().st_size
    if total is not None and final_size != total:
        raise OSError(
            f"download finished but size mismatch: {final_size} vs {total}")

    part_path.replace(dest_path)
    print(f"[download] complete: {dest_path} ({final_size / 1e9:.2f} GB)")
    return dest_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Download a DESI FITS catalog.")
    p.add_argument("--dataset", choices=["edr", "dr1"], default="edr",
                   help="edr (fuji, ~1 GB, for testing) or dr1 (iron, ~6-8 GB)")
    p.add_argument("--data-dir", default="data",
                   help="output directory (default: ./data)")
    args = p.parse_args()

    url = EDR_URL if args.dataset == "edr" else DR1_URL
    name = "zall-pix-fuji.fits" if args.dataset == "edr" else "zall-pix-iron.fits"
    ensure_fits(url, Path(args.data_dir) / name)
