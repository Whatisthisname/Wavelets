import numpy as np
import matplotlib.pyplot as plt
import trimesh
from helper import DAUBECHIES_a3, get_as, LOWER_a3, UPPER_a3


def cascade(as_: np.ndarray, j: int) -> np.ndarray:
    coefficients = np.array([1.0])

    for i in range(1, j + 1):
        result = np.zeros(len(coefficients) + (len(as_) - 1) * 2 ** (i - 1))

        for shift, a in enumerate(as_):
            shift_ = shift * 2 ** (i - 1)

            start = shift_
            end = shift_ + len(coefficients)

            result[start:end] += a * coefficients

        coefficients = result

    return coefficients


def build_grid(j=7, n_a3=80, a3_range=None):
    # recreate xs and a3s grid like in `ex5.py`
    # default a3 range: full valid interval from helper
    if a3_range is None:
        a3_min = LOWER_a3 + 0.0001
        a3_max = UPPER_a3 - 0.001
    else:
        a3_min, a3_max = a3_range

    # use a single cascade resolution as ex5
    # pick j the same as used during cascade
    sample_as = get_as(a3_min)
    resolution = 2 + 1 + (len(sample_as) - 1) * (2**j - 1)
    support_width = (resolution - 1) / 2**j
    xs = np.linspace(0, support_width, resolution)

    a3s = np.linspace(a3_min, a3_max, n_a3)

    # compute Z grid: shape (len(xs), len(a3s)) matching ex5's meshgrid(indexing='ij')
    Z = np.zeros((len(xs), len(a3s)))
    for i, a3 in enumerate(a3s):
        as_ = get_as(a3)
        result = np.concatenate(([0], cascade(as_, j), [0]))
        # result length should equal resolution
        if len(result) != len(xs):
            # if mismatch, resample or trim; here we'll pad or trim to fit xs
            if len(result) < len(xs):
                result = np.pad(result, (0, len(xs) - len(result)))
            else:
                result = result[: len(xs)]
        Z[:, i] = result

    X, Y = np.meshgrid(xs, a3s, indexing="ij")
    return X, Y, Z


def grid_to_trimesh(
    X,
    Y,
    Z,
    colormap_name="viridis",
    scale=(1.0, 10.0, 1.0),
    epsilon=0.01,
):
    """
    Convert a heightfield grid to a watertight mesh by
    - scaling X, Y, Z by `scale` (sx, sy, sz)
    - creating the top surface from Z
    - creating a matching flat bottom grid at base_z = min(Z) - epsilon
    - connecting the perimeter with vertical walls

    Returns a `trimesh.Trimesh` with vertex colors on the top surface (bottom/walls get the lowest color).
    """
    sx, sy, sz = scale
    nx, ny = X.shape

    # scale coordinates
    Xs = (X * sx).astype(float)
    Ys = (Y * sy).astype(float)
    Zs = (Z * sz).astype(float)

    top_verts = np.column_stack((Xs.ravel(), Ys.ravel(), Zs.ravel()))

    # top faces (same pattern as before)
    faces_top = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = i * ny + j
            v10 = (i + 1) * ny + j
            v01 = i * ny + (j + 1)
            v11 = (i + 1) * ny + (j + 1)
            faces_top.append([v00, v10, v11])
            faces_top.append([v00, v11, v01])

    faces_top = np.array(faces_top, dtype=np.int64)

    # compute base z a bit below the smallest Z
    z_min = Zs.min()
    z_max = Zs.max()
    base_z = z_min - abs(epsilon)

    # bottom verts: same X,Y grid but flat at base_z
    bottom_verts = np.column_stack((Xs.ravel(), Ys.ravel(), np.full(Xs.size, base_z)))

    # bottom faces: same triangulation but reversed orientation
    offset = top_verts.shape[0]
    faces_bottom = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            b00 = offset + i * ny + j
            b10 = offset + (i + 1) * ny + j
            b01 = offset + i * ny + (j + 1)
            b11 = offset + (i + 1) * ny + (j + 1)
            # reverse orientation so normal points down
            faces_bottom.append([b00, b11, b10])
            faces_bottom.append([b00, b01, b11])

    faces_bottom = np.array(faces_bottom, dtype=np.int64)

    # side faces: connect perimeter between top and bottom
    faces_sides = []
    # front/back along j=0 and j=ny-1
    for i in range(nx - 1):
        # front edge j=0
        t0 = i * ny + 0
        t1 = (i + 1) * ny + 0
        b0 = offset + t0
        b1 = offset + t1
        faces_sides.append([t0, t1, b1])
        faces_sides.append([t0, b1, b0])

        # back edge j=ny-1
        t0b = i * ny + (ny - 1)
        t1b = (i + 1) * ny + (ny - 1)
        b0b = offset + t0b
        b1b = offset + t1b
        faces_sides.append([t1b, t0b, b1b])
        faces_sides.append([t0b, b0b, b1b])

    # left/right along i=0 and i=nx-1 (skip corners duplication is okay; triangles will match)
    for j in range(ny - 1):
        # left i=0
        t0 = 0 * ny + j
        t1 = 0 * ny + (j + 1)
        b0 = offset + t0
        b1 = offset + t1
        faces_sides.append([t1, t0, b1])
        faces_sides.append([t0, b0, b1])

        # right i=nx-1
        t0r = (nx - 1) * ny + j
        t1r = (nx - 1) * ny + (j + 1)
        b0r = offset + t0r
        b1r = offset + t1r
        faces_sides.append([t0r, t1r, b1r])
        faces_sides.append([t0r, b1r, b0r])

    faces_sides = np.array(faces_sides, dtype=np.int64)

    verts = np.vstack((top_verts, bottom_verts))
    faces = np.vstack((faces_top, faces_bottom, faces_sides))

    # vertex colors: apply the colormap along the Y-axis (second axis)
    cmap = plt.get_cmap(colormap_name)

    # Use scaled Y values so the gradient runs along the Y direction
    top_y = Ys.ravel()
    bottom_y = Ys.ravel()
    all_y = np.hstack((top_y, bottom_y))

    y_min = top_y.min()
    y_max = top_y.max()
    y_range = max(y_max - y_min, 1e-12)

    ynorm_all = (all_y - y_min) / y_range
    ynorm_all = np.clip(ynorm_all, 0.0, 1.0)

    colors = (cmap(ynorm_all)[:, :3] * 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # attach vertex colors; use ColorVisuals when available
    try:
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    except Exception:
        # fallback
        mesh.visual.vertex_colors = colors

    # ensure watertight and clean normals
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.rezero()
    return mesh


def export_files(mesh: trimesh.Trimesh, out_prefix="wavelet_surface"):
    # export colored PLY (supports vertex colors) and STL (no color)
    ply_name = f"{out_prefix}.ply"
    stl_name = f"{out_prefix}.stl"
    gltf_name = f"{out_prefix}.glb"

    mesh.export(ply_name)
    print(f"Wrote {ply_name} (contains vertex colors)")

    # export STL (no color)
    mesh.export(stl_name)
    print(f"Wrote {stl_name} (no color - good for basic FDM printing)")

    # try glTF (binary) - some viewers and full-color printers accept glTF
    try:
        mesh.export(gltf_name)
        print(f"Wrote {gltf_name} (glTF binary, may preserve color in some pipelines)")
    except Exception:
        pass


if __name__ == "__main__":
    # Example: build a modest-size mesh and export
    X, Y, Z = build_grid(j=7, n_a3=80, a3_range=(DAUBECHIES_a3, 0.0))
    Z /= np.sqrt(2)
    eps = 1 - abs(Z.min())

    # example scales: width (x) in mm, depth (y) in mm, height (z) scaling
    mesh = grid_to_trimesh(
        X, Y, Z, colormap_name="viridis", scale=(1.0, 10.0, 1.0), epsilon=eps
    )
    export_files(mesh, out_prefix="wavelet_surface")
