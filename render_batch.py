#!/usr/bin/env python3
"""
Persistent Blender Cycles batch worker.
Renders a list of country .npz files in a single Blender process to avoid
startup overhead across ~200 countries.

Usage:
  blender --factory-startup --background --threads 3 --python render_batch.py -- <jobs.json>
"""
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1 :]
else:
    argv = []

if len(argv) < 1:
    print("Usage: blender --factory-startup --background --python render_batch.py -- <jobs.json|--stream> [METAL|CPU]")
    sys.exit(1)

MODE_ARG = argv[0]
DEVICE_ARG = argv[1].upper() if len(argv) > 1 else "METAL"

# Hypsometric color ramp matching the @cstats1 reference image (sRGB 0..1)
# Only the very top ~1.5% of peaks transition into snow white.
PALETTE_STOPS = np.array([
    [0.000, 0.34, 0.46, 0.59],  # slate / steel blue (lowlands)
    [0.180, 0.31, 0.51, 0.57],  # blue-teal
    [0.350, 0.35, 0.56, 0.47],  # muted sage / teal-green
    [0.520, 0.62, 0.67, 0.38],  # warm olive-green
    [0.680, 0.85, 0.70, 0.38],  # warm golden ochre
    [0.840, 0.90, 0.59, 0.46],  # warm salmon / peach
    [0.950, 0.91, 0.55, 0.43],  # rich warm coral-orange
    [0.985, 0.96, 0.85, 0.79],  # subtle rose-white tips (top ~1.5%)
    [1.000, 0.98, 0.98, 0.98],  # snow white summits
], dtype=np.float32)


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def eval_palette_linear(t):
    t = np.clip(t, 0.0, 1.0)
    stops = PALETTE_STOPS[:, 0]
    r = np.interp(t, stops, PALETTE_STOPS[:, 1])
    g = np.interp(t, stops, PALETTE_STOPS[:, 2])
    b = np.interp(t, stops, PALETTE_STOPS[:, 3])
    return srgb_to_linear(np.stack([r, g, b], axis=-1))


def configure_cycles(samples=48, res_x=1600, res_y=1320, use_metal=True):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if use_metal:
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "METAL"
            prefs.get_devices()
            gpu_found = False
            for d in prefs.devices:
                if d.type == "METAL":
                    d.use = True
                    gpu_found = True
                else:
                    d.use = False
            scene.cycles.device = "GPU" if gpu_found else "CPU"
        except Exception:
            scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.025
    scene.cycles.use_denoising = True
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene.cycles.max_bounces = 3
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 2

    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.15


def build_voxel_meshes(elev, z_height=1.35, color_ceil=1.0, z_base=0.07):
    H, W = elev.shape
    valid = ~np.isnan(elev)
    vals = elev[valid]

    e_min = float(np.percentile(vals, 0.5))
    e_max = float(np.percentile(vals, 99.95))
    if e_max <= e_min:
        e_max = e_min + 1.0

    norm = np.zeros_like(elev, dtype=np.float32)
    norm[valid] = np.clip((elev[valid] - e_min) / (e_max - e_min), 0.0, 1.0)
    color_t = norm * float(color_ceil)
    colors_lin = eval_palette_linear(color_t)

    scale = 10.0 / max(H, W)
    vr, vc = np.where(valid)
    cx = float(vc.mean())
    cy = float(vr.mean())

    z_grid = np.zeros((H, W), dtype=np.float32)
    z_grid[valid] = z_base + norm[valid] * z_height

    t_verts, t_faces, t_fcols = [], [], []
    s_verts, s_faces = [], []

    def add_quad(v_list, f_list, p0, p1, p2, p3):
        idx = len(v_list)
        v_list.extend([p0, p1, p2, p3])
        f_list.append((idx, idx + 1, idx + 2, idx + 3))

    for r in range(H):
        y1 = (cy - r) * scale
        y0 = y1 - scale
        for c in range(W):
            if not valid[r, c]:
                continue
            x0 = (c - cx) * scale
            x1 = x0 + scale
            zt = float(z_grid[r, c])
            col = (
                float(colors_lin[r, c, 0]),
                float(colors_lin[r, c, 1]),
                float(colors_lin[r, c, 2]),
                1.0,
            )
            add_quad(t_verts, t_faces, (x0, y0, zt), (x1, y0, zt), (x1, y1, zt), (x0, y1, zt))
            t_fcols.append(col)

            neighbors = [
                (0, -1, (x0, y1), (x0, y0)),
                (0, 1,  (x1, y0), (x1, y1)),
                (1, 0,  (x0, y0), (x1, y0)),
                (-1, 0, (x1, y1), (x0, y1)),
            ]
            for dr, dc, (ax, ay), (bx, by) in neighbors:
                nr, nc = r + dr, c + dc
                in_bounds = (0 <= nr < H) and (0 <= nc < W) and valid[nr, nc]
                if not in_bounds:
                    add_quad(s_verts, s_faces, (ax, ay, 0.0), (bx, by, 0.0), (bx, by, zt), (ax, ay, zt))
                else:
                    zn = float(z_grid[nr, nc])
                    if zn < zt - 1e-5:
                        add_quad(t_verts, t_faces, (ax, ay, zn), (bx, by, zn), (bx, by, zt), (ax, ay, zt))
                        t_fcols.append(col)

    t_mesh = bpy.data.meshes.new("TerrainMesh")
    t_mesh.from_pydata(t_verts, [], t_faces)
    t_mesh.update()
    col_attr = t_mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="CORNER")
    flat_cols = np.repeat(np.array(t_fcols, dtype=np.float32), 4, axis=0).ravel()
    col_attr.data.foreach_set("color", flat_cols)

    t_obj = bpy.data.objects.new("Terrain", t_mesh)
    bpy.context.collection.objects.link(t_obj)

    t_mat = bpy.data.materials.new("TerrainMat")
    t_mat.use_nodes = True
    nt = t_mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.82
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.18
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    t_obj.data.materials.append(t_mat)

    s_mesh = bpy.data.meshes.new("SkirtMesh")
    s_mesh.from_pydata(s_verts, [], s_faces)
    s_mesh.update()
    s_obj = bpy.data.objects.new("Skirt", s_mesh)
    bpy.context.collection.objects.link(s_obj)

    s_mat = bpy.data.materials.new("SkirtMat")
    s_mat.use_nodes = True
    s_bsdf = s_mat.node_tree.nodes.get("Principled BSDF")
    s_bsdf.inputs["Base Color"].default_value = (0.86, 0.86, 0.85, 1.0)
    s_bsdf.inputs["Roughness"].default_value = 0.88
    if "Specular IOR Level" in s_bsdf.inputs:
        s_bsdf.inputs["Specular IOR Level"].default_value = 0.10
    s_obj.data.materials.append(s_mat)

    xs_valid = (vc - cx) * scale
    ys_valid = (cy - vr) * scale
    bbox = (
        float(xs_valid.min()),
        float(xs_valid.max() + scale),
        float(ys_valid.min() - scale),
        float(ys_valid.max()),
        0.0,
        float(z_grid[valid].max()),
    )
    return bbox


def setup_studio_and_camera(bbox, cam_elev_deg=48.0, cam_azim_deg=0.0):
    scene = bpy.context.scene
    minx, maxx, miny, maxy, minz, maxz = bbox
    span_x = max(0.5, maxx - minx)
    span_y = max(0.5, maxy - miny)
    mid_x = 0.5 * (minx + maxx)
    mid_y = 0.5 * (miny + maxy)
    mid_z = 0.30 * maxz

    bpy.ops.mesh.primitive_plane_add(size=80.0, location=(mid_x, mid_y, 0.0))
    floor = bpy.context.active_object
    floor.name = "ShadowFloor"
    floor.is_shadow_catcher = True
    f_mat = bpy.data.materials.new("FloorMat")
    f_mat.use_nodes = True
    f_bsdf = f_mat.node_tree.nodes.get("Principled BSDF")
    f_bsdf.inputs["Base Color"].default_value = (0.97, 0.97, 0.97, 1.0)
    f_bsdf.inputs["Roughness"].default_value = 0.95
    floor.data.materials.append(f_mat)

    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    world.use_nodes = True
    w_bg = world.node_tree.nodes.get("Background")
    w_bg.inputs["Color"].default_value = (0.94, 0.96, 1.0, 1.0)
    w_bg.inputs["Strength"].default_value = 0.65

    sun_data = bpy.data.lights.new(name="KeySun", type="SUN")
    sun_data.energy = 4.0
    sun_data.color = (1.0, 0.985, 0.955)
    sun_data.angle = math.radians(6.0)
    sun_obj = bpy.data.objects.new(name="KeySun", object_data=sun_data)
    bpy.context.collection.objects.link(sun_obj)
    sun_dir = Vector((-0.78, -0.58, -0.68)).normalized()
    sun_obj.rotation_euler = sun_dir.to_track_quat("-Z", "Y").to_euler()

    fill_data = bpy.data.lights.new(name="FillLight", type="SUN")
    fill_data.energy = 0.35
    fill_data.color = (0.85, 0.92, 1.0)
    fill_data.angle = math.radians(35.0)
    fill_obj = bpy.data.objects.new(name="FillLight", object_data=fill_data)
    bpy.context.collection.objects.link(fill_obj)
    fill_dir = Vector((0.65, 0.45, -0.60)).normalized()
    fill_obj.rotation_euler = fill_dir.to_track_quat("-Z", "Y").to_euler()

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 75.0
    cam_data.clip_start = 0.1
    cam_data.clip_end = 500.0
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    elev_rad = math.radians(cam_elev_deg)
    azim_rad = math.radians(cam_azim_deg)
    target = Vector((mid_x - 0.03 * span_x, mid_y - 0.05 * span_y, mid_z))

    eff_w = span_x * 1.18
    eff_h = (span_y * math.sin(elev_rad) + maxz * math.cos(elev_rad)) * 1.38
    aspect = scene.render.resolution_x / scene.render.resolution_y
    sensor_w = cam_data.sensor_width
    sensor_h = sensor_w / aspect
    dist_w = (eff_w * 0.5) / (0.5 * sensor_w / cam_data.lens)
    dist_h = (eff_h * 0.5) / (0.5 * sensor_h / cam_data.lens)
    dist = max(dist_w, dist_h) * 1.08

    cam_offset = Vector((
        dist * math.cos(elev_rad) * math.sin(azim_rad),
        -dist * math.cos(elev_rad) * math.cos(azim_rad),
        dist * math.sin(elev_rad),
    ))
    cam_obj.location = target + cam_offset
    direction = (target - cam_obj.location).normalized()
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_one(job: dict):
    t0 = time.time()
    npz_path = Path(job["npz"])
    raw_png = Path(job["raw_png"])
    samples = int(job.get("samples", 48))
    res_x = int(job.get("res_x", 1600))
    res_y = int(job.get("res_y", 1320))

    data = np.load(npz_path, allow_pickle=True)
    elev = data["elevation"]
    z_height = float(data["z_height"])
    color_ceil = float(data["color_ceil"]) if "color_ceil" in data else 1.0
    cam_elev = float(data["cam_elev_deg"]) if "cam_elev_deg" in data else 48.0
    cam_azim = float(data["cam_azim_deg"]) if "cam_azim_deg" in data else 0.0

    bpy.ops.wm.read_factory_settings(use_empty=True)
    use_metal = job.get("device", DEVICE_ARG).upper() == "METAL"
    configure_cycles(samples=samples, res_x=res_x, res_y=res_y, use_metal=use_metal)
    bbox = build_voxel_meshes(elev, z_height=z_height, color_ceil=color_ceil, z_base=0.07)
    setup_studio_and_camera(bbox, cam_elev_deg=cam_elev, cam_azim_deg=cam_azim)

    raw_png.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(raw_png)
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    print(f"DONE_JOB {job['slug']} {dt:.1f}s", flush=True)


def main():
    if MODE_ARG == "--stream":
        print("READY", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line or line == "QUIT":
                break
            job = json.loads(line)
            try:
                render_one(job)
            except Exception as e:
                print(f"FAIL_JOB {job.get('slug', '?')} {e}", flush=True)
    else:
        jobs = json.loads(Path(MODE_ARG).read_text(encoding="utf-8"))
        for job in jobs:
            try:
                render_one(job)
            except Exception as e:
                print(f"FAIL_JOB {job['slug']} {e}", flush=True)


if __name__ == "__main__":
    main()
