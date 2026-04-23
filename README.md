# DESI Map Explorer

A GPU-accelerated 3D fly-through viewer for the [DESI](https://www.desi.lbl.gov/) galaxy redshift survey. Renders up to ~18 million galaxies and quasars in real-time, with point selection, six color modes, and direct links to external catalogs.

![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)

![DESI Map Explorer preview](assets/preview.png)

## Features

- **Fly-through camera** with WASD + mouse-look, camera roll, adjustable speed, and shift-boost
- **Point selection** (left-click) with camera lock-on, distance-scaled orbiting, unlimited pitch, and roll while locked
- **Info panel** for selected objects showing redshift, lookback time, distance, magnitudes, luminosity, g-r color, DESI target class, and RA/Dec
- **Clickable external links** for each selected object: Legacy Survey close-up, wide field, NED, SIMBAD
- **6 color modes** (press V to cycle):
  - **DESI target class** (default) - LRG / ELG / BGS / QSO / Secondary / Unclassified
  - **Redshift** (z) - cyan (nearby) to red (distant)
  - **Absolute magnitude** (M_r) - intrinsic brightness
  - **Spectral type** - GALAXY / QSO / other
  - **Rest-frame g-r** - blue (star-forming) to red (elliptical)
  - **Lookback time** - 0 Gyr (now) to 13 Gyr (ancient)
- **Color legend** (press G) with gradient bars or categorical swatches
- **Render distance** control ([ / ] keys) with GPU fog shader for depth cueing
- **3D distance grid** (press J) with concentric spheres at 1000/2000/4000/6000 Mpc and RGB axis lines
- **Two size modes** - absolute luminosity or apparent flux
- **LOD** - level-of-detail subset rendering for large datasets
- **Auto-download** - fetches the FITS catalog from DESI servers with resume support

## Quick Start

```bash
# Clone and set up
git clone https://github.com/itsRevela/DESI-DR1-Map-Explorer.git
cd DESI-DR1-Map-Explorer
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# Run with the EDR dataset (~2 GB download, ~1.2M points)
python main.py --dataset edr

# Run with the full DR1 dataset (~21 GB download, ~18M points)
python main.py --dataset dr1
```

The first run downloads the FITS file and builds a processed cache (`points_v4.npz`). Subsequent runs load the cache in seconds.

## Controls

| Key | Action |
|-----|--------|
| W / A / S / D | Move forward / left / back / right |
| F / C | Move up / down |
| Q / E | Roll left / right |
| Shift (hold) | 5x speed boost |
| Scroll wheel | Adjust movement speed |
| Right-click drag | Look around (cursor captured) |
| Left-click | Select / deselect nearest point |
| [ / ] | Decrease / increase render distance |
| V | Cycle color mode |
| G | Toggle color legend |
| J | Toggle distance grid |
| L | Toggle point size (luminosity / flux) |
| K | Toggle LOD on/off |
| T | Swap fly / turntable camera |
| H | Toggle help overlay |
| Esc | Quit |

When a point is selected, the camera locks onto it. WASD orbits around the target with distance-scaled speed. Q/E rolls the view. F/C orbits vertically with no pitch limit. Left-click again to deselect and resume free flight.

## Architecture

```
main.py          Entry point, CLI args, orchestrates pipeline
download.py      HTTP download with range-based resume + tqdm progress
process.py       FITS filtering, Planck18 cosmology, npz caching
viewer.py        Vispy + PyQt6 GPU renderer, camera, selection, UI
```

### Data Pipeline

1. **Download** - fetches the DESI zcatalog FITS file (~2-21 GB) with automatic resume
2. **Process** - filters for reliable extragalactic objects (ZWARN=0, not STAR, 0.001 < z < 4.0), computes Planck18 comoving distances, absolute magnitudes, lookback times, and caches everything to a `.npz` file
3. **View** - uploads positions to the GPU as a point cloud with additive blending and a custom depth fog shader for distance attenuation

### Selection & Picking

Point selection uses a KD-tree (built in background at startup) with ray-marching: 2000 samples along the view ray (hybrid log+linear spacing) query the tree for k=5 nearest neighbors. Candidates are filtered by render distance and angular threshold, then the closest to the camera wins. This scales to 18M+ points with ~1ms pick time.

## Data Sources

- [DESI Early Data Release (EDR)](https://data.desi.lbl.gov/public/edr/) - ~1.2M objects after filtering
- [DESI Data Release 1 (DR1)](https://data.desi.lbl.gov/public/dr1/) - ~18M objects after filtering
- Cosmology: [Planck 2018](https://docs.astropy.org/en/stable/cosmology/index.html) via astropy

## Requirements

- Python 3.14+
- GPU with OpenGL support (integrated graphics works for EDR; dedicated GPU recommended for DR1)
- ~500 MB RAM for EDR, ~4 GB for DR1
- See `requirements.txt` for Python packages

## License

This project uses publicly available DESI survey data. See the [DESI data policy](https://data.desi.lbl.gov/doc/policy/) for data usage terms.
