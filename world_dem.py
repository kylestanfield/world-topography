#!/usr/bin/env python3
"""
General all-countries DEM & boundary processor.
Handles every country on Earth cleanly:
  - Curated mainland bboxes for countries with distant overseas territories
  - Automatic cluster filtering for other MultiPolygon countries
  - Antimeridian-safe LAEA projection (Russia, Fiji, Kiribati, etc.)
  - Exact tile-set fetching (only downloads Terrarium tiles touched by land cells)
  - Dynamic zoom level (z=5 for continental, up to z=12 for microstates)
  - Realistic relief & color scaling for flat vs mountainous countries
"""
import concurrent.futures
import json
import math
import re
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer
from shapely import contains_xy
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.ops import transform as shapely_transform, unary_union

DATA_DIR = Path(__file__).parent / "data"
NPZ_DIR = DATA_DIR / "countries_npz"
TILES_DIR = DATA_DIR / "tiles"
NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
NE_PATH = DATA_DIR / "ne_10m_admin_0_countries.geojson"
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# Curated bounding boxes (min_lon, min_lat, max_lon, max_lat) for countries with
# distant overseas territories or special framing so the mainland looks iconic.
CURATED_BBOXES = {
    "United States of America": (-125.5, 24.0, -66.5, 49.5),
    "France": (-5.5, 41.0, 10.0, 51.5),
    "Netherlands": (3.1, 50.6, 7.4, 53.7),
    "Norway": (4.0, 57.5, 32.0, 71.5),
    "Denmark": (7.8, 54.4, 15.5, 57.9),
    "Spain": (-9.6, 35.8, 4.6, 44.0),
    "Portugal": (-9.7, 36.8, -6.0, 42.3),
    "United Kingdom": (-8.5, 49.8, 2.0, 61.0),
    "Japan": (128.0, 30.0, 146.5, 46.0),
    "Australia": (112.0, -44.0, 154.5, -10.0),
    "New Zealand": (166.0, -47.5, 179.0, -34.0),
    "Chile": (-76.0, -56.0, -66.0, -17.0),
    "Ecuador": (-81.5, -5.2, -75.0, 1.8),
    "South Africa": (16.0, -35.2, 33.2, -22.0),
    "Brazil": (-74.5, -34.0, -34.5, 5.5),
    "India": (68.0, 6.5, 97.5, 37.5),
    "Canada": (-141.5, 41.5, -52.0, 83.5),
}

# Friendly display titles where Natural Earth ADMIN is verbose
TITLE_OVERRIDES = {
    "United States of America": "Topography of the USA",
    "United Kingdom": "Topography of the United Kingdom",
    "Czechia": "Topography of Czechia",
    "United Republic of Tanzania": "Topography of Tanzania",
    "Democratic Republic of the Congo": "Topography of DR Congo",
    "Republic of the Congo": "Topography of the Republic of the Congo",
    "Republic of Serbia": "Topography of Serbia",
    "The Bahamas": "Topography of the Bahamas",
    "Gambia": "Topography of The Gambia",
    "Federated States of Micronesia": "Topography of Micronesia",
    "United Arab Emirates": "Topography of the UAE",
    "Central African Republic": "Topography of the Central African Republic",
    "Dominican Republic": "Topography of the Dominican Republic",
    "Philippines": "Topography of the Philippines",
    "Netherlands": "Topography of the Netherlands",
    "Maldives": "Topography of the Maldives",
    "Marshall Islands": "Topography of the Marshall Islands",
    "Solomon Islands": "Topography of the Solomon Islands",
    "Comoros": "Topography of the Comoros",
    "Seychelles": "Topography of the Seychelles",
}


def slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def ensure_natural_earth():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if NE_PATH.exists() and NE_PATH.stat().st_size > 1_000_000:
        return
    r = requests.get(NE_URL, timeout=60)
    r.raise_for_status()
    NE_PATH.write_bytes(r.content)


def load_all_countries(scope: str = "sovereign"):
    """Return list of dicts: {admin, slug, iso_a3, title, geom}."""
    ensure_natural_earth()
    with open(NE_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)

    # Group features by ADMIN name
    by_admin = {}
    for feat in gj["features"]:
        props = feat["properties"]
        ftype = props.get("TYPE", "")
        admin = props.get("ADMIN", "")
        sov = props.get("SOVEREIGNT", "")
        if not admin:
            continue
        if scope == "sovereign":
            # Include sovereign countries + sovereign core states (France, UK, USA, etc.)
            # plus widely mapped countries like Taiwan, Kosovo, Palestine
            is_sov = (ftype == "Sovereign country") or (
                ftype == "Country" and (admin == sov or admin in ("Taiwan", "Kosovo", "Palestine", "Greenland"))
            )
            if not is_sov:
                continue
        else:
            if ftype not in ("Sovereign country", "Country", "Dependency", "Disputed"):
                continue

        g = shape(feat["geometry"])
        if admin not in by_admin:
            by_admin[admin] = {
                "admin": admin,
                "slug": slugify(admin),
                "iso_a3": props.get("ADM0_A3") or props.get("ISO_A3") or slugify(admin).upper()[:3],
                "title": TITLE_OVERRIDES.get(admin, f"Topography of {admin}"),
                "geoms": [g],
            }
        else:
            by_admin[admin]["geoms"].append(g)

    countries = []
    for admin in sorted(by_admin.keys()):
        entry = by_admin[admin]
        entry["geom"] = unary_union(entry.pop("geoms"))
        countries.append(entry)
    return countries


def filter_primary_geometry(admin: str, geom):
    """Apply curated bbox or automatic outlier-island filter."""
    if admin in CURATED_BBOXES:
        clipped = geom.intersection(box(*CURATED_BBOXES[admin]))
        if not clipped.is_empty:
            return clipped

    if not isinstance(geom, MultiPolygon):
        return geom

    polys = list(geom.geoms)
    if len(polys) <= 1:
        return geom

    # Sort by approximate spherical area (cos(lat) weighted)
    def approx_area(p):
        c = p.centroid
        return p.area * max(0.1, math.cos(math.radians(c.y)))

    polys.sort(key=approx_area, reverse=True)
    main = polys[0]
    mc = main.centroid
    total_area = sum(approx_area(p) for p in polys)
    main_span_deg = max(main.bounds[2] - main.bounds[0], main.bounds[3] - main.bounds[1])

    # Keep polygons within a generous angular distance of the main polygon
    max_dist_deg = max(3.5, min(28.0, main_span_deg * 2.2))

    def ang_dist(p1, p2):
        dlon = abs(p1.x - p2.x)
        if dlon > 180.0:
            dlon = 360.0 - dlon
        dlat = abs(p1.y - p2.y)
        return math.hypot(dlon * math.cos(math.radians(0.5 * (p1.y + p2.y))), dlat)

    kept = [main]
    for p in polys[1:]:
        d = ang_dist(mc, p.centroid)
        if d <= max_dist_deg or (approx_area(p) / total_area >= 0.05 and d <= 45.0):
            kept.append(p)
    return unary_union(kept)


def circular_lon_center(geom) -> tuple[float, float]:
    """Compute antimeridian-safe (lon_c, lat_c) from polygon centroids weighted by area."""
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    sx, sy, slat, wsum = 0.0, 0.0, 0.0, 0.0
    for p in polys:
        c = p.centroid
        w = max(1e-9, p.area * max(0.1, math.cos(math.radians(c.y))))
        rad = math.radians(c.x)
        sx += w * math.cos(rad)
        sy += w * math.sin(rad)
        slat += w * c.y
        wsum += w
    lon_c = math.degrees(math.atan2(sy, sx))
    lat_c = slat / wsum
    return lon_c, lat_c


def lonlat_to_tile_xy(lon, lat, z):
    n = 2.0 ** z
    x = ((lon + 180.0) % 360.0) / 360.0 * n
    lat_rad = np.radians(np.clip(lat, -85.0511, 85.0511))
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(z: int, x: int, y: int):
    path = TILES_DIR / str(z) / str(x) / f"{y}.png"
    if path.exists() and path.stat().st_size > 100:
        return (z, x, y), path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = TERRARIUM_URL.format(z=z, x=x, y=y)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                path.write_bytes(r.content)
                return (z, x, y), path
        except Exception:
            if attempt == 2:
                raise
    return (z, x, y), None


def choose_zoom(cell_m: float, lat_c: float) -> int:
    """Pick Terrarium zoom level so tile pixel size is ~2-3x finer than voxel size."""
    # Equatorial meters per pixel at zoom z is 156543.03 / 2^z
    cos_lat = max(0.2, math.cos(math.radians(lat_c)))
    target_px_m = max(15.0, cell_m / 2.2)
    z = int(round(math.log2(156543.03 * cos_lat / target_px_m)))
    return max(5, min(12, z))


def build_country_npz(entry: dict, grid_max: int = 250, force: bool = False) -> Path:
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NPZ_DIR / f"{entry['slug']}.npz"
    if out_path.exists() and not force:
        return out_path

    geom_wgs = filter_primary_geometry(entry["admin"], entry["geom"])
    lon_c, lat_c = circular_lon_center(geom_wgs)

    proj_str = f"+proj=laea +lat_0={lat_c:.4f} +lon_0={lon_c:.4f} +datum=WGS84 +units=m"
    to_proj = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    to_wgs = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)

    geom_proj = shapely_transform(to_proj.transform, geom_wgs)
    if not geom_proj.is_valid:
        geom_proj = geom_proj.buffer(0)

    minx, miny, maxx, maxy = geom_proj.bounds
    width_m = max(1.0, maxx - minx)
    height_m = max(1.0, maxy - miny)

    if width_m >= height_m:
        W = grid_max
        H = max(24, int(round(grid_max * (height_m / width_m))))
    else:
        H = grid_max
        W = max(24, int(round(grid_max * (width_m / height_m))))

    dx_m = width_m / W
    dy_m = height_m / H
    cell_m = max(dx_m, dy_m)

    xs = minx + (np.arange(W) + 0.5) * dx_m
    ys = maxy - (np.arange(H) + 0.5) * dy_m
    XX, YY = np.meshgrid(xs, ys)

    mask = contains_xy(geom_proj, XX, YY)
    # Fallback for ultra-slender microstates if rasterization missed cells
    if mask.sum() < 12:
        geom_proj = geom_proj.buffer(cell_m * 0.6)
        mask = contains_xy(geom_proj, XX, YY)

    valid_idx = np.where(mask)
    vx = XX[valid_idx]
    vy = YY[valid_idx]

    z = choose_zoom(cell_m, lat_c)
    n_tiles = 2 ** z

    # Sample 5 points per voxel (center + 4 corners) and fetch ONLY tiles touched by land!
    offsets = [(0.0, 0.0), (-0.28, -0.28), (0.28, -0.28), (-0.28, 0.28), (0.28, 0.28)]
    sample_tx = []
    sample_ty = []
    unique_tiles = set()
    for ox, oy in offsets:
        lons, lats = to_wgs.transform(vx + ox * dx_m, vy + oy * dy_m)
        tx, ty = lonlat_to_tile_xy(lons, lats, z)
        sample_tx.append(tx)
        sample_ty.append(ty)
        ix = np.clip(np.floor(tx).astype(np.int32), 0, n_tiles - 1)
        iy = np.clip(np.floor(ty).astype(np.int32), 0, n_tiles - 1)
        for a, b in set(zip(ix.tolist(), iy.tolist())):
            unique_tiles.add((z, a, b))

    # Download required tiles concurrently
    tile_arrays = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(lambda c: fetch_tile(*c), unique_tiles))
    for key, p in results:
        if p is not None and p.exists():
            arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
            tile_arrays[(key[1], key[2])] = (
                arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0
            ) - 32768.0

    samples = np.zeros((len(offsets), len(vx)), dtype=np.float32)
    for i in range(len(offsets)):
        tx = sample_tx[i]
        ty = sample_ty[i]
        ix = np.clip(np.floor(tx).astype(np.int32), 0, n_tiles - 1)
        iy = np.clip(np.floor(ty).astype(np.int32), 0, n_tiles - 1)
        px = np.clip(((tx - ix) * 256.0).astype(np.int32), 0, 255)
        py = np.clip(((ty - iy) * 256.0).astype(np.int32), 0, 255)
        vals = np.zeros(len(vx), dtype=np.float32)
        for k, (tile_x, tile_y) in enumerate(zip(ix, iy)):
            t_arr = tile_arrays.get((int(tile_x), int(tile_y)))
            if t_arr is not None:
                vals[k] = t_arr[py[k], px[k]]
        samples[i] = vals

    sampled_elev = np.maximum(np.percentile(samples, 75, axis=0).astype(np.float32), 0.0)
    elev_grid = np.full((H, W), np.nan, dtype=np.float32)
    elev_grid[valid_idx] = sampled_elev

    # Adaptive relief height and color ceiling so flat countries stay realistic
    e_min = float(np.percentile(sampled_elev, 0.5))
    e_p99 = float(np.percentile(sampled_elev, 99.6))
    elev_span = max(1.0, e_p99 - e_min)

    # Full dramatic height (1.35) for countries with >= 1000m relief; gently scaled down for flat nations
    relief_factor = min(1.0, max(0.18, (elev_span / 900.0) ** 0.45))
    z_height = float(1.38 * relief_factor)

    # Color ceiling: if max elevation is tiny (< 400m), don't paint a 5m island snow-white!
    color_ceil = float(min(1.0, max(0.32, (elev_span / 600.0) ** 0.45)))

    np.savez_compressed(
        out_path,
        elevation=elev_grid,
        admin=entry["admin"],
        slug=entry["slug"],
        title=entry["title"],
        z_height=z_height,
        color_ceil=color_ceil,
        cam_elev_deg=48.0,
        cam_azim_deg=0.0,
        cell_size_m=cell_m,
        zoom=z,
        n_tiles=len(unique_tiles),
        elev_min=float(sampled_elev.min()),
        elev_max=float(sampled_elev.max()),
    )
    return out_path
