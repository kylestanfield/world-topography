# world-topography

Generates 3D voxel topography renders for every country using Python and Blender Cycles. Inspired by [@cstats1](https://x.com/cstats1)'s US topography map.

The full gallery is up on my blog at [kylestanfield.com/world-topography](https://kylestanfield.com/world-topography/).

## How it works

1. **`world_dem.py`** downloads Natural Earth 10m country borders and projects each country into a local Lambert Azimuthal Equal Area projection (centered on its circular mean longitude so high-latitude countries like Norway and antimeridian-crossing countries like Russia and Fiji don't break). It then grabs AWS Terrarium elevation tiles for only the tiles touched by land cells and saves a `.npz` height grid.
2. **`render_batch.py`** runs inside Blender (`--background`) as a persistent worker process. It builds a mesh of discrete square columns for each land cell, colors them with a hypsometric ramp, adds a white outer border skirt, and renders with Cycles on a shadow-catcher floor.
3. **`compose_poster.py`** composites the transparent Cycles render onto a white background, adds light grain, and draws the title.
4. **`run_all_countries.py`** ties it together with a two-ended queue: Metal GPU workers pull countries A-Z from the front while CPU workers pull Z-A from the back, skipping any country that already has a finished PNG.

## Setup

Requires Python 3.11+ and Blender (tested with Blender 5.2 on macOS / Apple Silicon).

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

Render all countries (resumable if interrupted):

```sh
.venv/bin/python run_all_countries.py
```

Render specific countries:

```sh
.venv/bin/python run_all_countries.py --countries Japan Switzerland Norway
```

Customize workers and sample count:

```sh
.venv/bin/python run_all_countries.py --gpu-workers 2 --cpu-workers 2 --cpu-threads 3 --samples 48
```

Finished posters are written to `output/countries/<country>.png` along with a local gallery at `output/index.html`.
