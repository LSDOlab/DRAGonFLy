"""
Write the FileWriter cases that test_filewriter.py replays through
ParaView's VTX reader.

usage: mpirun -n N python write_cases.py <outdir>

For every mesh and space, two writers (mesh_deformation False / True) write
three frames: the baseline mesh (Function path), deformation A (UFL
expression path) and deformation B (Function path). The deformations are
applied in place to geometry.x, ghosts included, as
Mesh.apply_node_motions does. One time-dependent writer per mesh labels its
frames with TIMES. Rank 0 writes manifest.json describing the files.
"""

import json
import os
import sys

import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

from dragonfly_sim.utils.filewriter import FileWriter

AMPLITUDES = (0.0, 0.1, -0.05)
TIMES = (0.0, 0.05, 0.1)


def deformation(x, amp):
    """Smooth node motion; test_filewriter.py applies the same formula to check the output."""
    d = np.zeros_like(x)
    d[:, 0] = amp * np.sin(np.pi * x[:, 1])
    d[:, 1] = 0.5 * amp * np.sin(np.pi * x[:, 0])
    return d


def meshes(comm):
    """Name -> mesh. The curved P2 mesh needs gmsh and is skipped without it."""
    out = {
        "quad2d": dolfinx.mesh.create_rectangle(
            comm, [[0, 0], [2, 1]], [16, 8], dolfinx.mesh.CellType.quadrilateral),
        "tri2d": dolfinx.mesh.create_rectangle(
            comm, [[0, 0], [2, 1]], [16, 8], dolfinx.mesh.CellType.triangle),
        "hex3d": dolfinx.mesh.create_box(
            comm, [[0, 0, 0], [1, 1, 1]], [5, 5, 5], dolfinx.mesh.CellType.hexahedron),
    }
    try:
        import gmsh
        from dolfinx.io import gmsh as gmshio
    except ImportError:
        return out
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if comm.rank == 0:
        outer = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        inner = gmsh.model.occ.addDisk(0, 0, 0, 0.3, 0.3)
        gmsh.model.occ.cut([(2, outer)], [(2, inner)])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [e[1] for e in gmsh.model.getEntities(2)], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
    out["tri2d_curved_p2"] = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2).mesh
    gmsh.finalize()
    return out


def spaces(gdim):
    """Name -> (element, expression builder). DG0, DG p>=1 and CG, scalar and vector."""
    def vec(x):
        return ufl.as_vector([ufl.sin(3 * x[0]) * x[1], ufl.cos(2 * x[1]), x[0] * x[1] ** 2]
                             + [1 + x[0] ** 3] * (gdim - 1))
    high = 3 if gdim == 2 else 2
    return {
        "dg0vec": (("DG", 0, (gdim + 2,)), vec),
        "dg0": (("DG", 0), lambda x: ufl.sin(3 * x[0]) * ufl.cos(2 * x[1])),
        "dg1vec": (("DG", 1, (gdim + 2,)), vec),
        "dg{}vec".format(high): (("DG", high, (gdim + 2,)), vec),
        "cg1vec": (("Lagrange", 1, (gdim,)),
                   lambda x: ufl.as_vector([x[i] ** 2 + x[(i + 1) % gdim] for i in range(gdim)])),
        "cg{}".format(high): (("Lagrange", high), lambda x: ufl.sin(3 * x[0]) * ufl.cos(2 * x[1])),
    }


def interpolated(V, expr):
    f = dolfinx.fem.Function(V)
    f.interpolate(dolfinx.fem.Expression(expr, V.element.interpolation_points))
    return f


def main(outdir):
    comm = MPI.COMM_WORLD
    manifest = {"amplitudes": AMPLITUDES, "times": TIMES, "files": []}
    for mesh_name, mesh in meshes(comm).items():
        gdim = mesh.geometry.dim
        x = ufl.SpatialCoordinate(mesh)
        x0 = mesh.geometry.x.copy()
        affine = mesh.geometry.cmaps[0].degree == 1

        writers = []
        for space_name, (element, build) in spaces(gdim).items():
            V = dolfinx.fem.functionspace(mesh, element)
            for deform in (False, True):
                name = "{}_{}_deform{}".format(mesh_name, space_name, int(deform))
                w = FileWriter(os.path.join(outdir, name), comm, V, mesh_deformation=deform)
                writers.append((w, V, build(x)))
                manifest["files"].append({
                    "name": name, "gathered": bool(w.gather), "deform": deform, "time": False,
                    # Output points are the geometry nodes (DG0 cell data on the
                    # mesh, CG1, DG1 on an affine mesh), so the deformation
                    # formula applies to them directly
                    "points_are_nodes": space_name in ("dg0vec", "dg0", "cg1vec")
                                        or (space_name == "dg1vec" and affine)})
        Vt = dolfinx.fem.functionspace(mesh, ("DG", 1, (gdim + 2,)))
        name = "{}_dg1vec_time".format(mesh_name)
        tw = FileWriter(os.path.join(outdir, name), comm, Vt, mesh_deformation=True,
                        time_dependent=True)
        texpr = spaces(gdim)["dg1vec"][1](x)
        manifest["files"].append({"name": name, "gathered": bool(tw.gather), "deform": True,
                                  "time": True, "points_are_nodes": affine})

        for k, amp in enumerate(AMPLITUDES):
            mesh.geometry.x[:] = x0 + deformation(x0, amp)
            for w, V, expr in writers:
                # Frame 1 goes through the UFL-expression path, the others
                # through the Function path
                w.interpolate_and_write(expr if k == 1 else interpolated(V, expr), write_counter=k)
            tw.interpolate_and_write(interpolated(Vt, texpr), time=TIMES[k])
        mesh.geometry.x[:] = x0
        del writers, tw

    comm.Barrier()
    if comm.rank == 0:
        with open(os.path.join(outdir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
