# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Shrink a world's object GLBs so the viewer loads them quickly, and give them
collision shapes.

    uv run scripts/compress_objects.py <world-slug> [--large id,id] [--objects id,id] [--dims dims.json]

Hunyuan ships every object with three 4096x4096 PNG textures (base colour,
normal, metallic-roughness): ~20 MB a model, ~380 MB for a 17-object room, and
about 85 MB of GPU memory per texture once decoded with mipmaps. A prop that
covers a few hundred pixels on screen needs none of that.

Textures are re-encoded as JPEG (core glTF, no extension needed) at 1024 px,
and 2048 for the base colour of large furniture. Large means a placed size of
at least LARGE_EXTENT_M in the world's scene.json (model extent x the viewer's
OBJECT_SCALE x the instance scale), or named with --large. Geometry is left
untouched.

Two hidden nodes are added, which the viewer uses instead of a bounding box:

* `collider-hull` -- points on the convex hull (a POINTS primitive), for props
  that move. Rapier builds the hull from these few hundred points instantly;
  from the raw 25-50k vertices it froze the page.
* `collider-mesh` -- the mesh simplified by vertex clustering on a grid, for
  furniture that does not move. The raw 50k-triangle mesh made every physics
  step test thousands of triangles under each prop resting on it.

With --dims (or `dimensions_m` on an object in image.json), the model is made
metric: every root node is wrapped in one `metric` node scaled so that, at the
viewer's OBJECT_SCALE and instance scale 1, the model's largest dimension is the
object's typical real size. A scene's instance scale then means "times life
size", and 1 is right. The factor is recorded in that node's extras
(`metricScale`, `largestM`). A scene authored against the unscaled model must
divide its scales by `metricScale`.

The original is kept under `worlds/<slug>/originals/<id>/` and is always the
input, so re-running never compresses twice.
"""

import argparse
import io
import json
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
WORLDS = REPO / "worlds"

PROP_SIZE = 1024
LARGE_BASE_COLOUR_SIZE = 2048
LARGE_EXTENT_M = 0.8
JPEG_QUALITY = 88
# Must match OBJECT_SCALE in app/src/modules/scene/SceneObject.tsx.
OBJECT_SCALE = 0.5

HULL_DIRECTIONS = 256
# Model units; at OBJECT_SCALE 0.5 and instance scale 1 this is ~3 cm.
CLUSTER_CELL = 0.06

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
COLLIDER_NODES = {"collider-hull", "collider-mesh"}
GLB_NAME = re.compile(r"^\d+-.+\.glb$")


def read_glb(data: bytes) -> tuple[dict, bytes]:
    magic, _, _ = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError("not a GLB")
    json_len, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == JSON_CHUNK
    gltf = json.loads(data[20:20 + json_len])
    offset = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", data, offset)
    assert bin_type == BIN_CHUNK
    return gltf, data[offset + 8:offset + 8 + bin_len]


def write_glb(gltf: dict, binary: bytes) -> bytes:
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<III", GLB_MAGIC, 2, total)
            + struct.pack("<II", len(js), JSON_CHUNK) + js
            + struct.pack("<II", len(binary), BIN_CHUNK) + binary)


def texture_roles(gltf: dict) -> dict[int, str]:
    """image index -> 'base' | 'normal' | 'other'."""
    roles: dict[int, str] = {}
    for material in gltf.get("materials", []):
        pbr = material.get("pbrMetallicRoughness", {})
        slots = [("base", pbr.get("baseColorTexture")),
                 ("normal", material.get("normalTexture")),
                 ("other", pbr.get("metallicRoughnessTexture")),
                 ("other", material.get("occlusionTexture")),
                 ("base", material.get("emissiveTexture"))]
        for role, ref in slots:
            if ref is None:
                continue
            source = gltf["textures"][ref["index"]].get("source")
            if source is not None:
                roles.setdefault(source, role)
    return roles


def encode(png: bytes, size: int) -> bytes:
    image = Image.open(io.BytesIO(png))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")  # metallic-roughness and normals carry no alpha
    if max(image.size) > size:
        image = image.resize((size, size), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=False)
    return out.getvalue()


def scene_vertices(gltf: dict, binary: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Every triangle of the default scene, in the scene root's frame."""
    def accessor(index: int) -> np.ndarray:
        acc = gltf["accessors"][index]
        view = gltf["bufferViews"][acc["bufferView"]]
        dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[acc["componentType"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride")
        item = np.dtype(dtype).itemsize * width
        if stride and stride != item:
            raw = np.frombuffer(binary, np.uint8, acc["count"] * stride, start).reshape(-1, stride)[:, :item]
            return raw.copy().view(dtype).reshape(-1, width)
        return np.frombuffer(binary, dtype, acc["count"] * width, start).reshape(-1, width)

    def local_matrix(node: dict) -> np.ndarray:
        if "matrix" in node:
            return np.array(node["matrix"], float).reshape(4, 4).T
        t = np.array(node.get("translation", [0, 0, 0]), float)
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        r = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        m = np.eye(4)
        m[:3, :3] = r * np.array(node.get("scale", [1, 1, 1]), float)
        m[:3, 3] = t
        return m

    verts, faces = [], []
    def walk(index: int, parent: np.ndarray) -> None:
        node = gltf["nodes"][index]
        world = parent @ local_matrix(node)
        if "mesh" in node:
            for prim in gltf["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                v = accessor(prim["attributes"]["POSITION"]).astype(float)
                v = (np.c_[v, np.ones(len(v))] @ world.T)[:, :3]
                f = accessor(prim["indices"]).reshape(-1, 3) if "indices" in prim else np.arange(len(v)).reshape(-1, 3)
                faces.append(f.astype(np.int64) + sum(len(x) for x in verts))
                verts.append(v)
        for child in node.get("children", []):
            walk(child, world)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        walk(root, np.eye(4))
    return np.concatenate(verts), np.concatenate(faces)


def hull_points(vertices: np.ndarray) -> np.ndarray:
    """Extreme points along evenly spread directions: a subset of the hull's
    vertices that spans it to within a sliver."""
    i = np.arange(HULL_DIRECTIONS) + 0.5
    phi = np.arccos(1 - 2 * i / HULL_DIRECTIONS)
    theta = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    extreme = np.unique(np.argmax(vertices @ dirs.T, axis=0))
    return vertices[extreme]


def cluster_mesh(vertices: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: merge vertices sharing a grid cell, drop collapsed and
    duplicate triangles."""
    keys = np.floor(vertices / cell + 0.5).astype(np.int64)
    _, remap, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    remap = remap.reshape(-1)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, remap, vertices)
    out_vertices = sums / counts[:, None]
    f = remap[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    return out_vertices.astype(np.float32), f[np.sort(keep)].astype(np.uint32)


def append_collider(gltf: dict, new_bin: bytearray, name: str, vertices: np.ndarray, faces: np.ndarray | None) -> None:
    """Add a hidden mesh node at the scene root carrying collision geometry."""
    def add_view(data: bytes, target: int) -> int:
        new_bin.extend(b"\0" * (-len(new_bin) % 4))
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(new_bin), "byteLength": len(data), "target": target})
        new_bin.extend(data)
        return len(gltf["bufferViews"]) - 1

    v = np.ascontiguousarray(vertices, np.float32)
    gltf["accessors"].append({
        "bufferView": add_view(v.tobytes(), 34962), "componentType": 5126, "count": len(v),
        "type": "VEC3", "min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist(),
    })
    primitive = {"attributes": {"POSITION": len(gltf["accessors"]) - 1}, "mode": 0}
    if faces is not None:
        idx = np.ascontiguousarray(faces.reshape(-1), np.uint32)
        gltf["accessors"].append({
            "bufferView": add_view(idx.tobytes(), 34963), "componentType": 5125, "count": len(idx), "type": "SCALAR",
        })
        primitive = {"attributes": primitive["attributes"], "indices": len(gltf["accessors"]) - 1, "mode": 4}
    gltf["meshes"].append({"name": name, "primitives": [primitive]})
    gltf["nodes"].append({"name": name, "mesh": len(gltf["meshes"]) - 1})
    gltf["scenes"][gltf.get("scene", 0)]["nodes"].append(len(gltf["nodes"]) - 1)


def compress(object_id: str, src: Path, dst: Path, placed_scale: float | None, force_large: bool,
             largest_m: float | None = None) -> tuple[int, int]:
    gltf, binary = read_glb(src.read_bytes())
    if any(node.get("name") in COLLIDER_NODES for node in gltf.get("nodes", [])):
        raise ValueError(f"{src} is already compressed; its original is missing")
    for img in gltf.get("images", []):
        if "uri" in img:
            raise ValueError(f"{src}: external image uri, not handled")

    vertices, faces = scene_vertices(gltf, binary)
    extent = float((vertices.max(axis=0) - vertices.min(axis=0)).max())
    metric = largest_m / (extent * OBJECT_SCALE) if largest_m else 1.0
    if largest_m:
        large = force_large or largest_m * (placed_scale or 1.0) >= LARGE_EXTENT_M
    else:
        large = force_large or (placed_scale is not None and extent * OBJECT_SCALE * placed_scale >= LARGE_EXTENT_M)

    roles = texture_roles(gltf)
    image_views = {img["bufferView"]: i for i, img in enumerate(gltf.get("images", []))}

    # Rebuild the binary chunk view by view, replacing image bytes.
    new_bin = bytearray()
    for view_index, view in enumerate(gltf["bufferViews"]):
        start = view.get("byteOffset", 0)
        chunk = binary[start:start + view["byteLength"]]
        if view_index in image_views:
            image_index = image_views[view_index]
            role = roles.get(image_index, "other")
            size = LARGE_BASE_COLOUR_SIZE if (large and role == "base") else PROP_SIZE
            chunk = encode(chunk, size)
            gltf["images"][image_index]["mimeType"] = "image/jpeg"
        new_bin += b"\0" * (-len(new_bin) % 4)  # every view stays 4-byte aligned
        view["byteOffset"] = len(new_bin)
        view["byteLength"] = len(chunk)
        new_bin += chunk

    hull = hull_points(vertices)
    # In model units, so a metric bake leaves colliders exactly as they were.
    mesh_v, mesh_f = cluster_mesh(vertices, faces, CLUSTER_CELL)
    append_collider(gltf, new_bin, "collider-hull", hull, None)
    append_collider(gltf, new_bin, "collider-mesh", mesh_v, mesh_f)
    if largest_m:
        scene = gltf["scenes"][gltf.get("scene", 0)]
        gltf["nodes"].append({"name": "metric", "scale": [metric] * 3, "children": scene["nodes"],
                              "extras": {"metricScale": round(metric, 6), "largestM": largest_m}})
        scene["nodes"] = [len(gltf["nodes"]) - 1]
    gltf["buffers"] = [{"byteLength": len(new_bin) + (-len(new_bin) % 4)}]

    out = write_glb(gltf, bytes(new_bin))
    dst.write_bytes(out)
    print(f"  {object_id:24s} {'large' if large else 'prop ':5s} hull {len(hull):3d} pts, "
          f"mesh {len(faces)} -> {len(mesh_f)} tris" + (f", metric x{metric:.3f} ({largest_m:.2f} m)" if largest_m else ""))
    return src.stat().st_size, len(out)


def placed_scales(world: Path) -> dict[str, float]:
    """Largest instance scale per object id, from the world's scene.json."""
    scene = world / "scene.json"
    if not scene.exists():
        return {}
    scales: dict[str, float] = {}
    for instance in json.loads(scene.read_text()).get("instances", []):
        parts = str(instance.get("assetId", "")).split("/")
        if len(parts) < 2:
            continue
        scale = max(instance.get("scale", [1, 1, 1]))
        scales[parts[1]] = max(scales.get(parts[1], 0.0), float(scale))
    return scales


def typical_sizes(world: Path, dims_path: str) -> dict[str, float]:
    """Largest typical dimension in metres per object id: from a dims file
    ({id: {height_m, width_m, depth_m}}), else from image.json's `dimensions_m`."""
    sizes: dict[str, float] = {}
    image = world / "image.json"
    if image.exists():
        for obj in json.loads(image.read_text()).get("objects", []):
            d = obj.get("dimensions_m")
            if d:
                sizes[obj["id"]] = max(float(d.get(k, 0)) for k in ("height_m", "width_m", "depth_m"))
    if dims_path:
        for oid, d in json.loads(Path(dims_path).read_text()).items():
            sizes[oid] = max(float(d[k]) for k in ("height_m", "width_m", "depth_m"))
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("world", help="world slug, a folder under worlds/")
    parser.add_argument("--large", default="", help="object ids to treat as large furniture, comma-separated")
    parser.add_argument("--objects", default="", help="only these object ids, comma-separated")
    parser.add_argument("--dims", default="", help="real-world dimensions per object id, JSON; makes models metric")
    args = parser.parse_args()

    world = WORLDS / args.world
    output = world / "output"
    if not output.is_dir():
        parser.error(f"{output} does not exist")
    force_large = {x for x in args.large.split(",") if x}
    only = {x for x in args.objects.split(",") if x}
    scales = placed_scales(world)
    sizes = typical_sizes(world, args.dims)

    before = after = 0
    for object_dir in sorted(p for p in output.iterdir() if p.is_dir() and p.name != "world"):
        object_id = object_dir.name
        if only and object_id not in only:
            continue
        for live in sorted(f for f in object_dir.iterdir() if GLB_NAME.match(f.name)):
            original = world / "originals" / object_id / live.name
            if not original.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(live, original)
            tmp = live.with_suffix(".glb.tmp")
            try:
                b, a = compress(object_id, original, tmp, scales.get(object_id), object_id in force_large,
                                sizes.get(object_id))
            except ValueError as error:
                tmp.unlink(missing_ok=True)
                print(f"  skipped {live.name}: {error}")
                continue
            tmp.replace(live)
            before += b
            after += a
    print(f"total {before / 1e6:.1f} MB -> {after / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
