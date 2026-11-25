#!/usr/bin/env python3
"""
Automated, resumable pipeline to generate 3D voxel topography posters for
every country on Earth.

Features:
  - Resumable/idempotent (skips already-rendered countries unless --force)
  - Parallel DEM prefetching + parallel persistent Blender Cycles workers
  - Live progress logging + manifest.json + searchable HTML gallery (output/index.html)

Examples:
  # Run all ~196 sovereign countries with 3 parallel Blender workers:
  .venv/bin/python run_all_countries.py

  # Test a handful of diverse edge-case countries first:
  .venv/bin/python run_all_countries.py --countries Italy Maldives Russia Fiji Monaco Iceland

  # Resume an interrupted run:
  .venv/bin/python run_all_countries.py
"""
import argparse
import collections
import concurrent.futures
import html
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from compose_poster import compose
from world_dem import build_country_npz, load_all_countries, slugify

MANIFEST_LOCK = threading.Lock()
QUEUE_LOCK = threading.Lock()

ROOT = Path(__file__).parent
RENDERS_DIR = ROOT / "renders" / "countries"
OUTPUT_DIR = ROOT / "output" / "countries"
MANIFEST_PATH = ROOT / "output" / "manifest.json"
GALLERY_PATH = ROOT / "output" / "index.html"


def write_gallery(manifest: dict):
    items = sorted(manifest.values(), key=lambda x: x["admin"])
    cards = []
    for it in items:
        if it.get("status") != "done":
            continue
        if not (OUTPUT_DIR / f"{it['slug']}.png").exists():
            continue
        rel = f"countries/{it['slug']}.png"
        admin = html.escape(it["admin"])
        title = html.escape(it["title"])
        emax = it.get("elev_max", 0)
        cards.append(
            f'<figure class="card" data-name="{admin.lower()}">'
            f'<a href="{rel}" target="_blank"><img loading="lazy" src="{rel}" alt="{title}"></a>'
            f'<figcaption><strong>{admin}</strong><span>max {emax:.0f} m</span></figcaption>'
            f"</figure>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Topography of the World ({len(cards)} Countries)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f6f6; color: #222; }}
  header {{ position: sticky; top: 0; z-index: 10; background: rgba(255,255,255,0.92); backdrop-filter: blur(8px); padding: 16px 24px; border-bottom: 1px solid #e5e5e5; display: flex; flex-wrap: wrap; gap: 16px; align-items: center; justify-content: space-between; }}
  h1 {{ margin: 0; font-family: Georgia, serif; font-style: italic; font-weight: 400; font-size: 1.5rem; }}
  input {{ padding: 8px 14px; font-size: 0.95rem; border: 1px solid #ccc; border-radius: 999px; width: 260px; outline: none; }}
  input:focus {{ border-color: #4e808e; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 18px; padding: 24px; max-width: 1800px; margin: 0 auto; }}
  .card {{ margin: 0; background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.06); transition: transform .15s; }}
  .card:hover {{ transform: translateY(-2px); }}
  .card img {{ width: 100%; display: block; aspect-ratio: 1600 / 1320; object-fit: cover; }}
  figcaption {{ padding: 10px 14px; display: flex; justify-content: space-between; font-size: 0.9rem; color: #444; }}
  figcaption span {{ color: #888; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<header>
  <h1>Topography of the World &mdash; {len(cards)} Countries</h1>
  <input id="q" type="search" placeholder="Filter countries..." oninput="filterCards(this.value)">
</header>
<main class="grid" id="grid">
{''.join(cards)}
</main>
<script>
function filterCards(q) {{
  q = q.trim().toLowerCase();
  for (const el of document.querySelectorAll('.card')) {{
    el.style.display = !q || el.dataset.name.includes(q) ? '' : 'none';
  }}
}}
</script>
</body>
</html>
"""
    GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    GALLERY_PATH.write_text(page, encoding="utf-8")


def run_stream_worker(
    label: str,
    device: str,
    threads: int,
    job_deque: collections.deque,
    pop_left: bool,
    manifest: dict,
    compose_pool: concurrent.futures.ThreadPoolExecutor,
):
    """Persistent Blender worker pulling jobs dynamically from left (A->Z) or right (Z->A)."""
    cmd = [
        "blender",
        "--factory-startup",
        "--background",
    ]
    if threads > 0:
        cmd.extend(["--threads", str(threads)])
    cmd.extend([
        "--python",
        str(ROOT / "render_batch.py"),
        "--",
        "--stream",
        device,
    ])
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def finish_compose(j, dt):
        compose(Path(j["raw_png"]), Path(j["out_png"]), j["title"])
        try:
            Path(j["raw_png"]).unlink(missing_ok=True)
        except Exception:
            pass
        with MANIFEST_LOCK:
            manifest[j["slug"]] = {
                "admin": j["admin"],
                "slug": j["slug"],
                "title": j["title"],
                "elev_max": j["elev_max"],
                "render_s": dt,
                "device": device,
                "status": "done",
            }
            MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            write_gallery(manifest)
            done_count = sum(1 for v in manifest.values() if v.get("status") == "done")
        print(f"[{label}] Finished {j['admin']} ({dt}) | total done: {done_count}", flush=True)

    # Wait for Blender READY signal
    for line in proc.stdout:
        if line.strip() == "READY":
            break

    while True:
        with QUEUE_LOCK:
            if not job_deque:
                break
            job = job_deque.popleft() if pop_left else job_deque.pop()
        job["device"] = device
        proc.stdin.write(json.dumps(job) + "\n")
        proc.stdin.flush()

        for line in proc.stdout:
            line = line.strip()
            if line.startswith("DONE_JOB "):
                _, slug, dt = line.split(" ", 2)
                compose_pool.submit(finish_compose, job, dt)
                break
            elif line.startswith("FAIL_JOB "):
                print(f"[{label}] {line}", flush=True)
                break

    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.stdin.close()
    except Exception:
        pass
    proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["sovereign", "all"], default="sovereign")
    parser.add_argument("--countries", nargs="*", help="Specific country names or slugs to render")
    parser.add_argument("--limit", type=int, default=0, help="Max number of countries to process")
    parser.add_argument("--gpu-workers", type=int, default=2, help="Metal GPU Blender workers (pop A->Z)")
    parser.add_argument("--cpu-workers", type=int, default=2, help="CPU Blender workers (pop Z->A)")
    parser.add_argument("--cpu-threads", type=int, default=3, help="CPU threads per CPU worker")
    parser.add_argument("--samples", type=int, default=48, help="Cycles samples")
    parser.add_argument("--res-x", type=int, default=1600)
    parser.add_argument("--res-y", type=int, default=1320)
    parser.add_argument("--grid-max", type=int, default=250)
    parser.add_argument("--force", action="store_true", help="Re-render even if output PNG exists")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    countries = load_all_countries(scope=args.scope)
    if args.countries:
        wanted = {slugify(c) for c in args.countries}
        countries = [
            c for c in countries
            if c["slug"] in wanted or c["admin"].lower() in {x.lower() for x in args.countries}
        ]
    if args.limit > 0:
        countries = countries[: args.limit]

    todo = []
    for c in countries:
        out_png = OUTPUT_DIR / f"{c['slug']}.png"
        if out_png.exists() and not args.force:
            if c["slug"] not in manifest:
                manifest[c["slug"]] = {
                    "admin": c["admin"],
                    "slug": c["slug"],
                    "title": c["title"],
                    "status": "done",
                }
            continue
        todo.append(c)

    print(
        f"Total matched: {len(countries)} | Already done: {len(countries) - len(todo)} | To render: {len(todo)}",
        flush=True,
    )
    if not todo:
        write_gallery(manifest)
        print(f"All requested countries already rendered! Gallery: {GALLERY_PATH}")
        return

    # Stage 1: Build DEM .npz files in parallel
    t_dem = time.time()
    print(f"Preparing DEM voxel grids for {len(todo)} countries...", flush=True)
    jobs = []

    def prep_one(c):
        npz = build_country_npz(c, grid_max=args.grid_max, force=args.force)
        import numpy as np
        d = np.load(npz, allow_pickle=True)
        return {
            "admin": c["admin"],
            "slug": c["slug"],
            "title": c["title"],
            "npz": str(npz),
            "raw_png": str(RENDERS_DIR / f"{c['slug']}_raw.png"),
            "out_png": str(OUTPUT_DIR / f"{c['slug']}.png"),
            "samples": args.samples,
            "res_x": args.res_x,
            "res_y": args.res_y,
            "elev_max": float(d["elev_max"]) if "elev_max" in d else 0.0,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(prep_one, c): c for c in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            c = futs[fut]
            try:
                j = fut.result()
                jobs.append(j)
                if i % 10 == 0 or i == len(todo):
                    print(f"  DEM ready {i}/{len(todo)} ({c['admin']})", flush=True)
            except Exception as e:
                print(f"  DEM FAILED for {c['admin']}: {e}", flush=True)

    print(f"Prepared {len(jobs)} DEM grids in {time.time() - t_dem:.1f}s", flush=True)

    # Stage 2: Sort alphabetically and launch GPU workers (A->Z) + CPU workers (Z->A)
    jobs.sort(key=lambda j: j["admin"])
    job_deque = collections.deque(jobs)
    t_ren = time.time()
    total_workers = args.gpu_workers + args.cpu_workers
    print(
        f"Launching {args.gpu_workers} Metal GPU workers (A->Z) + "
        f"{args.cpu_workers} CPU workers ({args.cpu_threads} threads each, Z->A)...",
        flush=True,
    )

    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=2) as compose_pool,
        concurrent.futures.ThreadPoolExecutor(max_workers=max(1, total_workers)) as worker_pool,
    ):
        futs = []
        for i in range(args.gpu_workers):
            futs.append(
                worker_pool.submit(
                    run_stream_worker, f"GPU-{i+1}", "METAL", 2, job_deque, True, manifest, compose_pool
                )
            )
        for i in range(args.cpu_workers):
            futs.append(
                worker_pool.submit(
                    run_stream_worker, f"CPU-{i+1}", "CPU", args.cpu_threads, job_deque, False, manifest, compose_pool
                )
            )
        for fut in futs:
            fut.result()

    write_gallery(manifest)
    print(f"All done in {(time.time() - t_ren) / 60:.1f} min! Gallery: {GALLERY_PATH}", flush=True)


if __name__ == "__main__":
    main()
