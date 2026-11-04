"""
All-hexahedral mesh around the BIG_FISH model (meshes/big_fish_domain.brep, made by fish_geometry.py): an extruded
wall-normal boundary layer (y+ ~ 1) and an unstructured hex mesh out to the R = 50 m sphere.  Run in the `dragonfly`
conda env.  The fish axis is z with the head at +z, so the wake extends towards -z.

OCC can neither offset nor fillet this CAD, so the boundary layer is built from the surface mesh:
  1. the skin is meshed with triangles; vertex normals are averaged, consistently oriented and smoothed;
  2. every vertex is extruded along its normal over a thickness T_i (BL_T, reduced locally wherever a prism would fold,
     e.g. in the concave fin root), with n layers whose growth per column keeps the first cell at Y1 (y+ = 1);
  3. the BL top surface is rebuilt as planar built-in-kernel triangles (one mesh triangle each) inside a faceted
     sphere, meshed with tets, and gmsh's hex subdivision (Mesh.SubdivisionAlgorithm = 2) splits tets into 4 hexes and
     each top triangle into 3 quads;
  4. below each top quad a column of hexes is added down to the wall; the wall nodes at edge midpoints and triangle
     centroids are projected onto the CAD.  The interface nodes are shared, so the mesh is conformal.

Usage:  python fish_unstructured.py [--brep ../../meshes/big_fish_domain.brep] [--out ../../meshes/big_fish_unstructured.msh]
"""
import argparse
import math
import sys
import time

import numpy as np
import gmsh
from scipy.spatial import cKDTree

import eel_blocking as eb

# --- flow / wall sizing (water, 20 C, 5 m/s, fish length 1.33 m)
U_INF, NU, LENGTH = 5.0, 1.0e-6, 1.33
RE_L = U_INF * LENGTH / NU
CF = 0.455 / math.log10(RE_L) ** 2.58
Y1 = NU / (U_INF * math.sqrt(CF / 2))          # first cell height for y+ = 1
R_FAR = 50.0
TAIL_Z = -0.6625                               # tail tip in the centred frame of big_fish_domain.brep
WAKE_LEN, WAKE_R = 3.4, 0.35


# ---------------------------------------------------------------- 1. skin surface mesh
def skin_mesh(brep, h_skin, h_min, curv, algo=6, h_strip=0.0015, strip_grade=0.08):
    """Triangles on the fish skin.  Returns nodes P, triangles T (consistently oriented, outward), the CAD face of each
    triangle, and the CAD projections of all edge midpoints and triangle centroids."""
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.model.add('skin')
    gmsh.model.occ.importShapes(brep)
    gmsh.model.occ.synchronize()
    far = [t for _, t in gmsh.model.getEntities(2) if max(abs(x) for x in gmsh.model.getBoundingBox(2, t)) > 5]
    gmsh.model.occ.remove(gmsh.model.getEntities(3))
    gmsh.model.occ.remove([(2, t) for t in far], recursive=True)
    gmsh.model.occ.synchronize()
    # faces separated by long chains of short CAD edges (the head/body junction is split into ~40 pieces per
    # quadrant) are meshed as one compound surface, so the chain does not force millimetre triangles
    pairs = {}
    for _, c in gmsh.model.getEntities(1):
        f = tuple(sorted(gmsh.model.getAdjacencies(1, c)[0]))
        if len(f) == 2:
            pairs.setdefault(f, []).append(c)
    group = {}
    for (a, b), cs in pairs.items():
        if len(cs) > 5:
            ga, gb = group.get(a, {a}), group.get(b, {b})
            g = ga | gb
            for t in g:
                group[t] = g
    for g in {frozenset(g) for g in group.values()}:
        gmsh.model.mesh.setCompound(2, sorted(g))
    # the fin's rounded edges are thin strips (~3 mm wide, ~1 mm radius): mesh them at ~1.5 mm (MeshAdapt; the
    # 'Unknown' offset surfaces give needles with Frontal-Delaunay) and grade to h_skin over ~80 mm
    strips = []
    for _, f in gmsh.model.getEntities(2):
        cs = [abs(c) for _, c in gmsh.model.getBoundary([(2, f)], oriented=False)]
        if gmsh.model.occ.getMass(2, f) / max(gmsh.model.occ.getMass(1, c) for c in cs) < 0.004:
            strips.append(f)
            gmsh.model.mesh.setAlgorithm(2, f, 1)
    fld = gmsh.model.mesh.field
    fld.add('Distance', 1); fld.setNumbers(1, 'SurfacesList', strips); fld.setNumber(1, 'Sampling', 30)
    fld.add('Threshold', 2); fld.setNumber(2, 'InField', 1)
    fld.setNumber(2, 'SizeMin', h_strip); fld.setNumber(2, 'SizeMax', h_skin)
    fld.setNumber(2, 'DistMin', 0.002); fld.setNumber(2, 'DistMax', strip_grade)
    fld.setAsBackgroundMesh(2)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    for k, v in dict(MeshSizeMax=h_skin, MeshSizeMin=min(h_min, h_strip), MeshSizeFromCurvature=curv, Algorithm=algo).items():
        gmsh.option.setNumber('Mesh.' + k, v)
    gmsh.model.mesh.generate(2)
    nid, xyz, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(nid)
    nid, X = nid[order], xyz.reshape(-1, 3)[order]
    tris, face = [], []
    for _, s in gmsh.model.getEntities(2):
        tg = gmsh.model.mesh.getElementsByType(2, s)[1]
        tris.append(np.asarray(tg).reshape(-1, 3)); face.append(np.full(len(tris[-1]), s))
    tris, face = np.vstack(tris), np.concatenate(face)
    used = np.unique(tris)
    P = X[np.searchsorted(nid, used)]
    T = np.searchsorted(used, tris)
    # merge nodes closer than 0.1 mm (CAD slivers) and drop the collapsed triangles
    rep = np.arange(len(P))
    for i, j in sorted(cKDTree(P).query_pairs(1e-4)):
        ri, rj = rep[i], rep[j]
        rep[rep == max(ri, rj)] = min(ri, rj)
    T = rep[T]
    keep = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    T, face = T[keep], face[keep]
    used = np.unique(T)
    P, T = P[used], np.searchsorted(used, T)
    T = orient(P, T)
    # CAD projections of edge midpoints and centroids
    E, tri_edges = edges_of(T)
    mid = 0.5 * (P[E[:, 0]] + P[E[:, 1]])
    cen = P[T].mean(1)
    # closest points can land on the untrimmed extension of a BSpline face: a projection is accepted only if it lies
    # inside the face and moves the point by less than 0.3 x the local edge length; otherwise the chord point is kept
    def project(pts, faces, lim):
        best, dbest = pts.copy(), np.full(len(pts), np.inf)
        for col in range(faces.shape[1]):
            for s in np.unique(faces[:, col]):
                m = faces[:, col] == s
                q = np.asarray(gmsh.model.getClosestPoint(2, int(s), pts[m].ravel())[0]).reshape(-1, 3)
                d = np.linalg.norm(q - pts[m], axis=1)
                inside = np.array([gmsh.model.isInside(2, int(s), list(x)) > 0 for x in q])
                take = inside & (d < lim[m]) & (d < dbest[m])
                idx = np.where(m)[0][take]
                best[idx], dbest[idx] = q[take], d[take]
        return best
    edge_faces = np.zeros((len(E), 2), int)
    edge_faces[tri_edges.ravel()[::-1], 1] = np.repeat(face, 3)[::-1]
    edge_faces[tri_edges.ravel(), 0] = np.repeat(face, 3)
    mid_p = project(mid, edge_faces, 0.3 * np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1))
    cen_p = project(cen, face[:, None], 0.3 * np.linalg.norm(P[T] - np.roll(P[T], 1, 1), axis=2).min(1))
    gmsh.finalize()
    return P, T, E, tri_edges, mid_p, cen_p


def edges_of(T):
    """Unique edges E (sorted pairs) and, per triangle, the index of edges (ab, bc, ca)."""
    e = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1).reshape(-1, 2)
    E, inv = np.unique(np.sort(e, 1), axis=0, return_inverse=True)
    return E, inv.reshape(-1, 3)


def orient(P, T):
    """Make the triangle orientation consistent over the closed skin and outward (positive enclosed volume)."""
    e = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1).reshape(-1, 2)
    key = np.sort(e, 1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 3)
    owners = {}
    for t in range(len(T)):
        for j in range(3):
            owners.setdefault(int(inv[t, j]), []).append((t, j))
    flip = np.zeros(len(T), bool); seen = np.zeros(len(T), bool)
    for start in range(len(T)):
        if seen[start]:
            continue
        seen[start] = True; stack = [start]
        while stack:
            t = stack.pop()
            for j in range(3):
                a, b = T[t, j], T[t, (j + 1) % 3]
                if flip[t]:
                    a, b = b, a
                for u, jj in owners[int(inv[t, j])]:
                    if u == t or seen[u]:
                        continue
                    c, d = T[u, jj], T[u, (jj + 1) % 3]
                    flip[u] = (c == a and d == b)     # the neighbour must run the shared edge the other way
                    seen[u] = True; stack.append(u)
    T = np.where(flip[:, None], T[:, ::-1], T)
    vol = np.einsum('ij,ij->i', P[T[:, 0]], np.cross(P[T[:, 1]], P[T[:, 2]])).sum() / 6
    return T if vol > 0 else T[:, ::-1]


# ---------------------------------------------------------------- 2. boundary-layer columns
def vertex_normals(P, T, n_smooth):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    fn = np.cross(b - a, c - a)
    fu = fn / np.linalg.norm(fn, axis=1, keepdims=True)
    N = np.zeros_like(P)
    for j in range(3):                                         # angle-weighted average
        u = P[T[:, (j + 1) % 3]] - P[T[:, j]]; v = P[T[:, (j + 2) % 3]] - P[T[:, j]]
        ang = np.arccos(np.clip(np.einsum('ij,ij->i', u, v) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1))
        np.add.at(N, T[:, j], fu * ang[:, None])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    E, _ = edges_of(T)
    for _ in range(n_smooth):                                  # Laplacian smoothing of the normal field
        S = np.zeros_like(N); np.add.at(S, E[:, 0], N[E[:, 1]]); np.add.at(S, E[:, 1], N[E[:, 0]])
        N = N + 0.5 * S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-30)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
    for _ in range(200):                                       # visibility: n . n_face >= 0.3 for all incident faces
        d = np.stack([np.einsum('ij,ij->i', N[T[:, j]], fu) for j in range(3)], 1)
        if d.min() >= 0.3:
            break
        for j in range(3):
            m = d[:, j] < 0.3
            np.add.at(N, T[m, j], 0.2 * fu[m])
        N /= np.linalg.norm(N, axis=1, keepdims=True)
    return N, fu


def prism_ok(P, T, top, fu, smin=0.05):
    """Per triangle: the 3 hexes per layer that the subdivision makes below it (vertex, edge midpoints, centroid) have
    scaled corner Jacobians > smin at every height through the boundary layer."""
    D = top - P
    def pts(s):
        V = P + s * D
        a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        return a, b, c, 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a), (a + b + c) / 3
    da, db, dc = D[T[:, 0]], D[T[:, 1]], D[T[:, 2]]
    dirs = (da, db, dc, 0.5 * (da + db), 0.5 * (db + dc), 0.5 * (dc + da), (da + db + dc) / 3)
    quads = ((0, 3, 6, 5), (1, 4, 6, 3), (2, 5, 6, 4))           # (vertex, mid, centroid, mid), outward oriented
    ok = np.ones(len(T), bool)
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        X = pts(s)
        for q in quads:
            for j in range(4):
                p0, p1, p3 = X[q[j]], X[q[(j + 1) % 4]], X[q[(j - 1) % 4]]
                d = dirs[q[j]]
                e1, e3 = p1 - p0, p3 - p0
                J = np.einsum('ij,ij->i', np.cross(e1, e3), d)
                ok &= J > smin * np.linalg.norm(e1, axis=1) * np.linalg.norm(e3, axis=1) * np.linalg.norm(d, axis=1)
    return ok


def column_sj(cols, quads):
    """Minimum scaled corner Jacobian over all layers of the hex columns below each quad (cols: (nc, n+1, 3))."""
    Q = cols[quads]                                             # (nq, 4, n+1, 3)
    D = Q[:, :, 1:] - Q[:, :, :-1]                              # column edges per layer
    sj = np.full(len(quads), np.inf)
    for j in range(4):
        for lev in (0, 1):                                      # bottom and top face of each layer
            X = Q[:, :, :-1] if lev == 0 else Q[:, :, 1:]
            e1, e3, d = X[:, (j + 1) % 4] - X[:, j], X[:, (j - 1) % 4] - X[:, j], D[:, j]
            J = np.einsum('qkx,qkx->qk', np.cross(e1, e3), d)
            J /= np.linalg.norm(e1, axis=2) * np.linalg.norm(e3, axis=2) * np.linalg.norm(d, axis=2)
            sj = np.minimum(sj, J.min(1))
    return sj


def bl_thickness(P, T, N, fu, T0, slope=0.5, iters=60):
    """Thickness per vertex: T0, reduced where prisms fold, made Lipschitz (|dT| <= slope * distance)."""
    Ti = np.full(len(P), T0)
    E, _ = edges_of(T)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    for it in range(iters):
        bad = ~prism_ok(P, T, P + Ti[:, None] * N, fu)
        if not bad.any():
            break
        Ti[np.unique(T[bad])] *= 0.8
        for _ in range(20):                                   # propagate: T_i <= T_j + slope * |x_i - x_j|
            old = Ti.copy()
            np.minimum.at(Ti, E[:, 0], Ti[E[:, 1]] + slope * L)
            np.minimum.at(Ti, E[:, 1], Ti[E[:, 0]] + slope * L)
            if np.array_equal(old, Ti):
                break
    return Ti, it, int((~prism_ok(P, T, P + Ti[:, None] * N, fu)).sum())


def column_fractions(Ti, n):
    """Cumulative heights (n+1 per vertex) with first cell Y1 and per-column geometric growth."""
    cum = np.zeros((len(Ti), n + 1))
    for i, t in enumerate(Ti):
        q = eb.line_q(t, n, Y1)
        h = Y1 * q ** np.arange(n) if q > 1 else np.full(n, t / n)
        cum[i, 1:] = np.cumsum(h) * t / h.sum()
    return cum


# ---------------------------------------------------------------- 3. outer region (geo kernel, faceted boundaries)
def icosphere(R, level):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(level):
        cache, F2 = {}, []
        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in cache:
                m = V[i] + V[j]; V.append(m / np.linalg.norm(m)); cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    return R * np.array(V), np.array(F)


def faceted_surfaces(geo, P, T, lc):
    """One planar triangle surface per mesh triangle; each constrained to a single mesh triangle."""
    pt = [geo.addPoint(*p, lc[i]) for i, p in enumerate(P)]
    E, te = edges_of(T)
    ln = [geo.addLine(pt[a], pt[b]) for a, b in E]
    surf = []
    for t, (a, b, c) in enumerate(T):
        loop = []
        for j, (u, v) in enumerate(((a, b), (b, c), (c, a))):
            e = te[t, j]
            loop.append(ln[e] if E[e, 0] == u else -ln[e])
        surf.append(geo.addPlaneSurface([geo.addCurveLoop(loop)]))
    return pt, ln, surf


def size_fields(top_pts, args):
    f = gmsh.model.mesh.field
    f.add('Distance', 1); f.setNumbers(1, 'PointsList', top_pts)
    f.add('MathEval', 2); f.setString(2, 'F', f'Min({args.h_far}, {args.h_iface} + {args.grow} * F1)')
    z0 = TAIL_Z + 0.3                                         # wake tube from the fin root to WAKE_LEN behind the tail
    f.add('MathEval', 3)
    f.setString(3, 'F', f'{args.h_wake} + 0.3 * (Max(0, Sqrt(x*x + y*y) - {WAKE_R}) + '
                        f'Max(0, {TAIL_Z - WAKE_LEN} - z) + Max(0, z {-z0:+.4f}))')
    f.add('Min', 4); f.setNumbers(4, 'FieldsList', [2, 3])
    f.setAsBackgroundMesh(4)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--brep', default='../../meshes/big_fish_domain.brep')
    ap.add_argument('--out', default='../../meshes/big_fish_unstructured.msh')
    ap.add_argument('--h-skin', type=float, default=0.03, help='max triangle size on the skin')
    ap.add_argument('--h-min', type=float, default=0.003, help='min triangle size on the skin')
    ap.add_argument('--h-strip', type=float, default=0.0025, help='triangle size on the rounded fin edges')
    ap.add_argument('--strip-grade', type=float, default=0.025, help='distance over which h_strip grows to h_skin')
    ap.add_argument('--curv', type=float, default=0, help='elements per 2 pi of curvature on the skin')
    ap.add_argument('--bl-t', type=float, default=0.015, help='boundary-layer thickness [m]')
    ap.add_argument('--q-bl', type=float, default=1.25, help='wall-normal growth for the full-thickness columns')
    ap.add_argument('--h-iface', type=float, default=0.05)
    ap.add_argument('--grow', type=float, default=0.35)
    ap.add_argument('--h-wake', type=float, default=0.07)
    ap.add_argument('--h-far', type=float, default=12.0)
    ap.add_argument('--sphere-level', type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()

    P, T, E, te, mid_p, cen_p = skin_mesh(args.brep, args.h_skin, args.h_min, args.curv, 6, args.h_strip,
                                          args.strip_grade)
    print(f'skin: {len(P)} nodes, {len(T)} triangles  ({time.time() - t0:.0f} s)', flush=True)
    N, fu = vertex_normals(P, T, n_smooth=10)
    n = eb.graded(args.bl_t, Y1, args.q_bl)['n']
    Ti, it, nbad = bl_thickness(P, T, N, fu, args.bl_t)
    print(f'BL: y+=1 first cell {Y1 * 1e6:.2f} um, {n} layers, thickness {Ti.min() * 1e3:.2f}..{Ti.max() * 1e3:.1f} mm '
          f'({(Ti < 0.999 * args.bl_t).sum()} of {len(Ti)} columns reduced, {it} iterations, {nbad} folded prisms left)',
          flush=True)
    if nbad:
        sys.exit('boundary-layer prisms still fold; lower --bl-t or refine the skin')
    cum = column_fractions(Ti, n)                              # (nv, n+1)
    colv = P[:, None, :] + cum[:, :, None] * N[:, None, :]       # vertex columns (nv, n+1, 3)
    s_v = cum / Ti[:, None]
    # edge-midpoint and centroid columns: averages, with the wall node moved onto the CAD (correction fades upward)
    colm = 0.5 * (colv[E[:, 0]] + colv[E[:, 1]])
    colm += (mid_p - colm[:, 0])[:, None, :] * (1 - 0.5 * (s_v[E[:, 0]] + s_v[E[:, 1]]))[:, :, None]
    colc = colv[T].mean(1)
    colc += (cen_p - colc[:, 0])[:, None, :] * (1 - s_v[T].mean(1))[:, :, None]
    cols = np.concatenate([colv, colm, colc])                   # all columns; layer n is the BL top
    chord = np.concatenate([colv, 0.5 * (colv[E[:, 0]] + colv[E[:, 1]]), colv[T].mean(1)])
    # the 3 quads per skin triangle that the subdivision will produce, in column indices (outward oriented)
    nv, ne = len(P), len(E)
    m_ab, m_bc, m_ca, cc = nv + te[:, 0], nv + te[:, 1], nv + te[:, 2], nv + ne + np.arange(len(T))
    sub = np.vstack([np.stack([T[:, 0], m_ab, cc, m_ca], 1), np.stack([T[:, 1], m_bc, cc, m_ab], 1),
                     np.stack([T[:, 2], m_ca, cc, m_bc], 1)])
    for it in range(10):                                        # drop CAD projections that make bad wall hexes
        sj = column_sj(cols, sub)
        bad = sj < 0.05
        if not bad.any():
            break
        fix = np.unique(sub[bad][:, 1:])
        fix = fix[np.any(cols[fix] != chord[fix], axis=(1, 2))]
        if not len(fix):
            break
        cols[fix] = chord[fix]
    print(f'BL hexes: min scaled corner Jacobian {sj.min():.3f}, {int((sj < 0.05).sum())} wall quads below 0.05 '
          f'({it} projection-revert passes)', flush=True)
    top = cols[:, -1]

    # ---- outer region
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.model.add('fish')
    geo = gmsh.model.geo
    Etop = np.linalg.norm(top[E[:, 0]] - top[E[:, 1]], axis=1)
    lc = np.full(len(P), np.inf); np.minimum.at(lc, E[:, 0], Etop); np.minimum.at(lc, E[:, 1], Etop)
    tp, tl, ts = faceted_surfaces(geo, colv[:, -1], T, lc)
    SP, ST = icosphere(R_FAR, args.sphere_level)
    sp, sl, ss = faceted_surfaces(geo, SP, ST, np.full(len(SP), args.h_far))
    vol = geo.addVolume([geo.addSurfaceLoop(ss), geo.addSurfaceLoop(ts)])
    for c in tl + sl:
        geo.mesh.setTransfiniteCurve(c, 2)
    for s in ts + ss:
        geo.mesh.setTransfiniteSurface(s)
    geo.synchronize()
    size_fields(tp, args)
    for k, v in dict(MeshSizeFromPoints=1, MeshSizeFromCurvature=0, MeshSizeExtendFromBoundary=1, RecombineAll=0,
                     Algorithm=6, Algorithm3D=1, SubdivisionAlgorithm=2, SecondOrderLinear=1, Optimize=1,
                     OptimizeNetgen=1).items():
        gmsh.option.setNumber('Mesh.' + k, v)
    gmsh.model.mesh.generate(3)
    ty, tg, _ = gmsh.model.mesh.getElements(3, vol)
    print(f'outer region: {dict(zip(map(int, ty), map(len, tg)))} (5 = hex)  ({time.time() - t0:.0f} s)', flush=True)

    # ---- BL hexes under the split top quads
    quads = np.vstack([np.asarray(gmsh.model.mesh.getElementsByType(3, s)[1]).reshape(-1, 4) for s in ts])
    qn = np.unique(quads)
    qx = np.array([gmsh.model.mesh.getNode(int(t))[0] for t in qn])
    d, col_of = cKDTree(top).query(qx)
    assert d.max() < 1e-7, f'top node matching failed ({d.max():.2e} m)'
    qcol = col_of[np.searchsorted(qn, quads)]                  # column index per quad corner
    # orientation: bottom face (wall) normal must point into the BL
    b = cols[qcol[:, 0], 0]; w1 = cols[qcol[:, 1], 0] - b; w3 = cols[qcol[:, 3], 0] - b
    up = cols[qcol[:, 0], 1] - b
    flip = np.einsum('ij,ij->i', np.cross(w1, w3), up) < 0
    quads = np.where(flip[:, None], quads[:, [0, 3, 2, 1]], quads)
    qcol = np.where(flip[:, None], qcol[:, [0, 3, 2, 1]], qcol)
    ucols = np.unique(qcol)
    nxt = int(gmsh.model.mesh.getMaxNodeTag()) + 1
    tag = np.zeros((len(cols), n + 1), np.int64)
    tag[ucols, :n] = nxt + np.arange(len(ucols) * n).reshape(len(ucols), n)
    tag[col_of, n] = qn
    newtags = tag[ucols, :n].ravel()
    newxyz = cols[ucols, :n].reshape(-1, 3)
    bl = gmsh.model.addDiscreteEntity(3)
    gmsh.model.mesh.addNodes(3, bl, newtags, newxyz.ravel())
    H = tag[qcol]                                               # (nq, 4, n+1)
    hexa = np.concatenate([H[:, :, :-1], H[:, :, 1:]], axis=1).transpose(0, 2, 1).reshape(-1, 8)
    e0 = int(gmsh.model.mesh.getMaxElementTag()) + 1
    gmsh.model.mesh.addElementsByType(bl, 5, np.arange(e0, e0 + len(hexa)), hexa.ravel())
    body = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addElementsByType(body, 3, np.arange(e0 + len(hexa), e0 + len(hexa) + len(H)),
                                      H[:, ::-1, 0].ravel())
    gmsh.model.addPhysicalGroup(2, [body], name='body')
    gmsh.model.addPhysicalGroup(2, ss, name='farfield')
    gmsh.model.addPhysicalGroup(3, [vol, bl], name='fluid')
    print(f'BL: {len(hexa)} hexes on {len(H)} wall quads  ({time.time() - t0:.0f} s)', flush=True)
    report(body, ss, P, T)
    gmsh.model.mesh.renumberNodes()      # dolfinx's reader needs node tags 1..N without gaps
    gmsh.model.mesh.renumberElements()
    gmsh.option.setNumber('Mesh.MshFileVersion', 4.1)
    gmsh.write(args.out)
    gmsh.finalize()


def report(body, far, P, T):
    etags, enodes = gmsh.model.mesh.getElementsByType(5)
    sj = np.array(gmsh.model.mesh.getElementQualities(list(etags), 'minSJ'))
    nid, nc, _ = gmsh.model.mesh.getNodes()
    X = np.zeros((int(nid.max()) + 1, 3)); X[nid] = nc.reshape(-1, 3)
    hx = X[np.asarray(enodes).reshape(-1, 8)]
    ed = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    el = np.stack([np.linalg.norm(hx[:, a] - hx[:, b], axis=1) for a, b in ed], 1)
    ar = el.max(1) / el.min(1)
    print(f'hexahedra: {len(etags)}   min scaled Jacobian {sj.min():.3f}   1% quantile {np.quantile(sj, .01):.3f}   '
          f'<0.2: {(sj < .2).sum()}   <=0: {(sj <= 0).sum()}')
    print(f'edge aspect ratio: median {np.median(ar):.1f}  99% {np.quantile(ar, .99):.0f}  max {ar.max():.0f}')
    vol = np.array(gmsh.model.mesh.getElementQualities(list(etags), 'volume'))
    faces = np.sort(np.asarray(gmsh.model.mesh.getElementFaceNodes(5, 4)).reshape(-1, 4), 1)
    ev = np.repeat(np.arange(len(etags)), 6)
    order = np.lexsort(faces.T[::-1]); key, ev = faces[order], ev[order]
    same = np.all(key[1:] == key[:-1], 1)
    h = np.cbrt(np.abs(vol)); a, b = ev[:-1][same], ev[1:][same]
    jump = np.maximum(h[a], h[b]) / np.minimum(h[a], h[b])
    print(f'neighbour size jump: median {np.median(jump):.2f}  99% {np.quantile(jump, .99):.2f}  max {jump.max():.2f}')
    _, cnt = np.unique(key, axis=0, return_counts=True)
    nb = sum(len(gmsh.model.mesh.getElements(2, t)[1][0]) for t in [body] + list(far))
    print(f'conformity: {int((cnt == 1).sum())} free faces vs {nb} boundary quads; faces shared by >2 hexes: {int((cnt > 2).sum())}')
    print(f'total volume {vol.sum():.3f} m3')
    c = hx.mean(1)
    print('wake (|x|,|y| < 0.25 m behind the tail): max edge length [cm] per window of distance behind the tail tip')
    for d0, d1 in ((0, .5), (.5, 1), (1, 2), (2, 3), (3, 4), (4, 6)):
        m = (c[:, 2] < TAIL_Z - d0) & (c[:, 2] >= TAIL_Z - d1) & (np.abs(c[:, 0]) < .25) & (np.abs(c[:, 1]) < .25)
        if m.any():
            print(f'  {d0:.1f}-{d1:.1f} m: {m.sum():6d} cells  median {np.median(el[m].max(1)) * 100:5.1f}  '
                  f'max {el[m].max() * 100:5.1f}')


if __name__ == '__main__':
    sys.exit(main())
