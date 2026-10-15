# Meshes

This page describes what DRAGonFLy needs from a mesh, how meshes are read, and the two ways of telling the solver
which boundaries are the wall, the symmetry plane and the far field.

## What the solver needs
- **Cells**: a DOLFINx mesh, quadrilaterals in 2D and hexahedra in 3D in all the examples. The mesh must be
  ghosted across shared facets, which every reader below does (the interior-facet terms need it).
- **Orientation**: the free stream flows in $+x$ at zero angle of attack. In 2D, y is vertical. In 3D, y is spanwise
  and z vertical, and the model is a half-wing.
- **Three kinds of boundary**: every exterior facet of the mesh belongs to exactly one of
  - the **wall**, the body (slip wall for Euler, adiabatic no-slip wall for RANS). It is the surface the shape
    parameterization moves.
  - a **symmetry plane**, treated as a slip wall. Only 3D half-wing meshes have one.
  - the **far field**, which gets the characteristic (Riemann-invariant) boundary condition (see
    [far fields](flow_models.md)).

  The far field is never specified: it is everything that is neither wall nor symmetry plane. Telling the solver
  which facets are the wall, and which are symmetry, is the "boundary type definition" below.

## Reading a mesh
`load_dolfinx_mesh` reads a mesh file, chosen by its extension, and returns `(mesh, info)`:

| Extension | Format | Notes |
|---|---|---|
| `.xdmf` | DOLFINx XDMF/HDF5 | `mesh_name` is the grid name (default `"mesh"`). Boundary tags: the grid `"Facet tags"`, or the one named by `facet_tags_name`. |
| `.msh` | gmsh | `tdim` is the mesh dimension (2 or 3). Boundary tags: the physical groups of the facets, with their names. |
| `.cgns` | structured multi-block CGNS | `tdim=2` for a grid one cell thick, `tdim=3` for a volume grid. Boundary tags: the `Wall` (1) and `Farfield` (2) families of a 2D grid. A volume grid has none. |

```python
from dragonfly_sim.utils.mesh_io_utils import load_dolfinx_mesh

mesh, info = load_dolfinx_mesh("naca0012.xdmf")
mesh, info = load_dolfinx_mesh("wing.msh", tdim=3)
mesh, info = load_dolfinx_mesh("OAT15A_Rizzi_1.cgns")        # info: chord, cell and boundary-edge counts
mesh, info = load_dolfinx_mesh("wing_vol_L3.cgns", tdim=3)   # info: point, cell and block counts
```

A 2D CGNS grid must be one cell thick (or planar). Rank 0 reads it with PyVista, keeps one span plane of every
block, merges the nodes the blocks share by exact coordinate match, and checks that the boundary edges are exactly
those of the wall and far-field families (`wall_family="Wall"`, `farfield_family="Farfield"`). By default the points
are shifted and scaled so that the wall's leading edge is at $x = 0$ and its chord is 1 (`scale_to_chord`). A 3D
grid becomes hexahedra, the blocks' shared nodes are merged by exact coordinate match, and the mesh is checked for
inverted cells and unmerged block interfaces. VTK does not read its boundary conditions. DOLFINx then distributes
the mesh; no converted file is written.

A mesh can also be read with plain DOLFINx, for example
`dolfinx.io.XDMFFile(comm, "naca0012.xdmf", "r").read_mesh()`. Such a mesh carries no boundary tags unless you attach
them (see below), so it can only be used with the boundary function.

### Boundary tags on the mesh
Many mesh generators tag the boundaries (gmsh physical groups, CGNS families, an XDMF facet-tag grid).
`load_dolfinx_mesh` keeps these tags and stores them on the mesh it returns:

| Attribute | Content |
|---|---|
| `mesh.facet_tags` | a DOLFINx `MeshTags` on the facets, or `None` if the file has none |
| `mesh.facet_tag_names` | `{name: tag value}` for the tags that have a name (gmsh, CGNS), otherwise `{}` |

The solver looks for these two attributes when it is given the mesh. For a mesh read some other way, set them with
`attach_facet_tags(mesh, facet_tags, tag_names=None)` from the same module, where `facet_tags` is a `MeshTags` of the
facets, for example from `xdmf.read_meshtags(mesh, "Facet tags")`.

## Defining the boundary types
`DG_windtunnel_model` (and `Mesh`, which it builds) finds the wall in one of two ways, and optionally the symmetry
planes. Either way, all other boundary facets are far field.

### Option 1: a boundary function
`mesh_inner_bdry_function` takes the midpoints of the exterior facets, as an array of shape `(dim, n)`, and returns
a boolean array of length `n`: `True` for the wall.

```python
def wall(x):                                     # x[0] = x, x[1] = y (, x[2] = z)
    return (x[0] > -0.01) & (x[0] < 1.01) & (x[1] > -0.1) & (x[1] < 0.1)

model = DG_windtunnel_model(mesh, boundary_dict, mesh_inner_bdry_function=wall, poly_order=0)
```
`dragonfly_sim.utils.mesh_manager_utils` provides `airfoil_inner_bdry_function` and `wing_inner_bdry_function` for the
meshes used in the examples. The function needs no tags, so it works with every mesh. In 3D, the symmetry plane is
fixed at $y = 0$: the exterior facets with midpoint $y \le 10^{-6}$.

### Option 2: the mesh's boundary tags
`wall_tags` names the tags that are the wall, and `symmetry_tags` the tags that are symmetry planes. Each is a tag
value, a tag name (where the format has names), or a list of either.

```python
mesh, _ = load_dolfinx_mesh("wing.msh", tdim=3)       # physical groups "wing", "root", "farfield"
model = DG_windtunnel_model(mesh, boundary_dict, shape, wall_tags="wing", symmetry_tags="root", poly_order=0)

mesh, _ = load_dolfinx_mesh("naca0012.xdmf")          # facet tags 2 (wall) and 3 (far field)
model = DG_windtunnel_model(mesh, boundary_dict, wall_tags=2, poly_order=0)
```
Only exterior facets are considered; tagged interior facets are ignored. Facets with a tag that is in neither list,
and untagged exterior facets, are far field. With tags, the symmetry planes are exactly the `symmetry_tags` facets,
in 2D or 3D, instead of the fixed $y = 0$ plane.

The same options exist in `examples/flow_analysis.py`: `--wall-tags 2`, and `--symmetry-tags` for a symmetry plane.

### Which one is used
1. `mesh_inner_bdry_function`, if given. `wall_tags` and `symmetry_tags` are then ignored.
2. Otherwise `wall_tags`.
3. Otherwise the FFD block of the shape parameterization: if it has block corners, its wall function is used (see
   [shape parameterization](shape_parameterization.md)).

If none applies, the model raises an error.

**Warning when tags are available.** If the function is used (explicitly, or through the FFD block) and the mesh also
carries boundary tags, a `UserWarning` says so and lists the tags with the number of exterior facets each marks,
and the name where there is one. For the NACA0012 mesh of the examples:

```text
The input mesh has boundary facet tags: 2 (240 facets), 3 (408 facets). The boundary function is used to find the
wall, so the boundary facet tags are currently unused. To use the tags instead, leave out the boundary function and
pass wall_tags (the tags of the body) and, if there is a symmetry plane, symmetry_tags; all other boundaries are far
field.
```
With `wall_tags` also given, the message adds that `wall_tags` and `symmetry_tags` are ignored. The warning appears
once per run (from rank 0).

**Errors.** A mesh that has tags, but is given neither a function nor `wall_tags`, raises an error that lists the
tags. `wall_tags` or `symmetry_tags` that are not on the mesh's exterior facets (or a name the mesh does not have),
a tag that is both wall and symmetry, or `wall_tags` on a mesh without tags, also raise.

### The tags inside the solver
Whichever option is used, `set_up_sim` turns the result into the solver's own facet tags: wall 2
(`model.WALL_TAG`), symmetry 5 (`model.SYMMETRY_TAG`) and far field 6 (`model.FARFIELD_TAG`). These are independent
of the tag values of the input mesh. The model prints the number of wall, symmetry and far-field facets per MPI rank
and writes `output_meshtags.xdmf`, so the assignment can be checked in ParaView. The mesh warper holds the far
field fixed, and the wall is the surface the shape parameterization moves.

### Tagging a mesh
For the tag option, the tags must mark the boundary facets, not only the cells:
- **gmsh**: give the boundary curves (2D) or surfaces (3D) physical groups, and name them. Name the cell group too, or
  gmsh writes only the tagged entities. Write the mesh as `.msh` (format 4.1).
- **XDMF**: write the facet `MeshTags` to the same file as the mesh, named `"Facet tags"`:

  ```python
  with dolfinx.io.XDMFFile(comm, "mesh.xdmf", "w") as xdmf:
      xdmf.write_mesh(mesh)
      facet_tags.name = "Facet tags"
      xdmf.write_meshtags(facet_tags, mesh.geometry)
  ```
