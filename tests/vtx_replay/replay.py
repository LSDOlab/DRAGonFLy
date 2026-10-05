"""
Rebuild the grid ParaView's VTX reader produces from a .bp file, and measure
it.

``vtxdump.cpp`` reads one step of a VTX file the way ParaView's reader does
(all writer blocks concatenated, connectivity offset per block, shared points
not merged) and dumps the raw arrays. The functions here build a
vtkUnstructuredGrid from that dump and report the quantities that show MPI
partition seams: duplicated points and the number of exterior edges (2D) or
faces (3D) that ParaView's Surface representation would draw.
"""

import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def build_vtxdump(build_dir):
    """Compile vtxdump.cpp against the ADIOS2 of this Python environment.

    Returns the executable path, or None when no C++ compiler or no ADIOS2
    C++ headers/libraries are available.
    """
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    prefix = sys.prefix
    if (cxx is None or not os.path.isfile(os.path.join(prefix, "include", "adios2.h"))
            or not any(f.startswith("libadios2_cxx") for f in os.listdir(os.path.join(prefix, "lib")))):
        return None
    exe = os.path.join(build_dir, "vtxdump")
    cmd = [cxx, "-std=c++17", "-O1", os.path.join(HERE, "vtxdump.cpp"),
           "-I" + os.path.join(prefix, "include"), "-L" + os.path.join(prefix, "lib"),
           "-ladios2_cxx", "-ladios2_core", "-Wl,-rpath," + os.path.join(prefix, "lib"),
           "-o", exe]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return exe


def dump(vtxdump, bp_file, out_dir, step):
    """Dump step `step` of `bp_file` into `out_dir`; returns `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([vtxdump, bp_file, out_dir, str(step)], check=True,
                   capture_output=True, text=True)
    return out_dir


def _read(d):
    meta = [line.split() for line in open(os.path.join(d, "meta.txt"))]
    info = {m[0]: m[1:] for m in meta if m[0] not in ("f64", "u8")}
    arrays = {m[1]: (m[0], int(m[2]), int(m[3])) for m in meta if m[0] in ("f64", "u8")}
    xml = open(os.path.join(d, "vtk.xml")).read()
    cell_names = (re.findall(r'Name="([^"]+)"', xml.split("<CellData>")[1])
                  if "<CellData>" in xml else [])
    geom = np.fromfile(os.path.join(d, "geometry.bin"), dtype=np.float64).reshape(-1, 3)
    conn = np.fromfile(os.path.join(d, "connectivity.bin"), dtype=np.int64)
    return info, arrays, cell_names, geom, conn


def load(d):
    """The dumped step as a dict: geometry, connectivity, field values and metadata."""
    info, arrays, cell_names, geom, conn = _read(d)
    ncells = int(info["connectivity"][0])
    kind, n, ncol = arrays["solution"]
    values = np.fromfile(os.path.join(d, "solution.bin"), dtype=np.float64).reshape(n, ncol)
    return {"blocks": int(info["nblocks"][0]),
            "vtk_type": int(info["type"][0]),
            "time": float(info["stepvalue"][0]),
            "geometry": geom,
            "connectivity": conn,
            "cells": conn.reshape(ncells, -1)[:, 1:],
            "values": values,
            "cell_data": "solution" in cell_names}


def summary(d):
    """Blocks, points, unique points, cells and exterior edges (2D) / faces (3D)."""
    import vtk
    from vtk.util import numpy_support as ns

    data = load(d)
    geom = data["geometry"]
    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    points.SetData(ns.numpy_to_vtk(geom, deep=True))
    grid.SetPoints(points)
    cell_array = vtk.vtkCellArray()
    cell_array.ImportLegacyFormat(ns.numpy_to_vtkIdTypeArray(data["connectivity"], deep=True))
    grid.SetCells(data["vtk_type"], cell_array)

    surface = vtk.vtkGeometryFilter()
    surface.SetInputData(grid)
    surface.Update()
    if grid.GetCell(0).GetCellDimension() == 3:
        exterior = surface.GetOutput().GetNumberOfCells()
    else:
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(surface.GetOutput())
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.Update()
        exterior = edges.GetOutput().GetNumberOfCells()

    return {"blocks": data["blocks"],
            "points": geom.shape[0],
            "unique_points": len(np.unique(np.round(geom, 10), axis=0)),
            "cells": grid.GetNumberOfCells(),
            "exterior": exterior}


def values_by_cell(d):
    """Field values keyed by rounded cell centroid, so dumps of different rank counts compare.

    Cell data gives one row per cell; point data gives the cell's nodal rows
    sorted by node coordinate.
    """
    data = load(d)
    geom, cells, values = data["geometry"], data["cells"], data["values"]
    out = {}
    for c, nodes in enumerate(cells):
        key = tuple(np.round(geom[nodes].mean(axis=0), 8))
        if data["cell_data"]:
            out[key] = values[c][None, :]
        else:
            rows = np.hstack([np.round(geom[nodes], 8), values[nodes]])
            out[key] = rows[np.lexsort(rows[:, :3].T[::-1])][:, 3:]
    return out


def max_value_difference(d_a, d_b):
    """Max |difference| of the field between two dumps of the same mesh."""
    a, b = values_by_cell(d_a), values_by_cell(d_b)
    if set(a) != set(b):
        raise AssertionError("the two dumps hold different cells: {} vs {}".format(len(a), len(b)))
    return max(float(np.max(np.abs(a[k] - b[k]))) for k in a)
