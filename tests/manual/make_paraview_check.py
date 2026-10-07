"""
Write a VTX file set for checking FileWriter output by hand in ParaView.

usage (from the repository root, meshes in meshes/):
    OMP_NUM_THREADS=1 mpirun -n 8 python tests/manual/make_paraview_check.py \\
        [--reference-filewriter path/to/other/filewriter.py]

Writes a new folder vtx_paraview_check_np<N>_<timestamp>_<random>/ in the
current directory with, for the NACA0012 mesh (DG0, DG1, DG2) and the STW L3
wing (DG0, DG1):

- <mesh>_DG<p>.bp: a smooth multi-component field written by FileWriter;
- <mesh>_DG<p>_reference.bp: the same field written by the FileWriter in
  --reference-filewriter (e.g. an older version), for a side-by-side check;
- <mesh>_partition_np<N>.bp: the owning MPI rank of each cell (DG0).

Open the files with the ADIOS2VTXReader. Seams show up as:

- 2D: Extract Surface, then Feature Edges with only Boundary Edges checked:
  lines along the partition boundaries (compare with the partition file);
- smooth colouring: Cell Data to Point Data, then Contour on a component:
  contours kinked or broken at partition boundaries;
- 3D: Surface representation at low opacity: internal partition walls.

DG p >= 1 output has separate points for every cell, so Feature Edges shows
every cell edge, in serial as well as in parallel.
"""

import argparse
import importlib.util
import os
import secrets
import time

import dolfinx
import ufl
from mpi4py import MPI

from dragonfly_sim.utils.filewriter import FileWriter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASES = (("naca0012", "meshes/naca0012_euler_mesh_quad_v2.xdmf", "mesh", (0, 1, 2)),
         ("stw_wing_L3", "meshes/wing_vol_L3_xdmf.xdmf", "Grid", (0, 1)))


def load_writer(path):
    spec = importlib.util.spec_from_file_location("reference_filewriter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FileWriter


def flow_like_field(mesh):
    """Smooth, multi-component field with gradients concentrated near the body."""
    gdim = mesh.geometry.dim
    x = ufl.SpatialCoordinate(mesh)
    bump = ufl.exp(-sum(x[i] ** 2 for i in range(gdim)))
    comps = [1 + 0.5 * bump * ufl.sin(3 * x[0]),
             bump * ufl.cos(2 * x[1]) * ufl.sin(2 * x[0]),
             0.3 * bump * ufl.sin(4 * x[0] * x[1])]
    if gdim == 3:
        comps.append(0.3 * bump * ufl.cos(3 * x[2]))
    comps.append(1 + 0.4 * bump * ufl.cos(3 * x[0] + 2 * x[1]))
    return ufl.as_vector(comps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--reference-filewriter", default=None)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    reference = load_writer(args.reference_filewriter) if args.reference_filewriter else None

    outdir = None
    if comm.rank == 0:
        outdir = "vtx_paraview_check_np{}_{}_{}".format(
            comm.size, time.strftime("%Y%m%d-%H%M%S"), secrets.token_hex(3))
        os.makedirs(outdir)
    outdir = comm.bcast(outdir, root=0)

    for tag, mesh_file, grid, degrees in CASES:
        with dolfinx.io.XDMFFile(comm, os.path.join(REPO, mesh_file), "r") as f:
            mesh = f.read_mesh(name=grid)
        gdim = mesh.geometry.dim
        expr = flow_like_field(mesh)
        for p in degrees:
            V = dolfinx.fem.functionspace(mesh, ("DG", p, (gdim + 2,)))
            u = dolfinx.fem.Function(V)
            u.interpolate(dolfinx.fem.Expression(expr, V.element.interpolation_points))
            writers = [("", FileWriter)] + ([("_reference", reference)] if reference else [])
            for suffix, Writer in writers:
                w = Writer(os.path.join(outdir, "{}_DG{}{}".format(tag, p, suffix)), comm, V)
                w.interpolate_and_write(u, write_counter=0)
                del w

        V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
        rank = dolfinx.fem.Function(V0)
        rank.x.array[:] = comm.rank
        w = FileWriter(os.path.join(outdir, "{}_partition_np{}".format(tag, comm.size)), comm, V0)
        w.interpolate_and_write(rank, write_counter=0)
        del w

    if comm.rank == 0:
        print("wrote", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
