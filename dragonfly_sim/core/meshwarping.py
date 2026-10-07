from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

import resource
import datetime
import os
from time import perf_counter

# These only take effect if jax's backend has not been initialised yet; export
# them before starting Python to be sure. Note that XLA:CPU sizes its own
# thread pool from NPROC (else the core count), which neither setting bounds:
# measured 2026-09-30 on jax 0.11.2, each rank uses ~3.4 cores in the warp
# kernels without NPROC. A one-worker pool (NPROC=1, or one-core pinning) is
# NOT set here: under one-core pinning, nested XLA executions from inside csdl
# host callbacks deadlocked.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import jax
# Mandatory, and before any idwarp_jax call: the kd-tree warper takes its host
# dtype from jnp.asarray(volume_points), so without x64 it silently runs in
# float32 -- which cannot difference O(300) coordinates against O(1e-5) motions.
jax.config.update("jax_enable_x64", True)

from idwarp_jax import build_volume_pts_func
from idwarp_jax.warp import compute_idwarp_reference_length

import csdl_alpha as csdl
# ordered_callbacks needs csdl_alpha with ordered custom-operation callbacks (branch
# mpi_callback_issue); older versions ignore it, and MPI runs give wrong gradients.
assert hasattr(csdl.CustomExplicitOperation, "ordered_callbacks"), \
    "csdl_alpha is too old: custom-operation callbacks are not ordered under MPI"

# build_volume_pts_func applies a rigid pitch rotation to its output; pitch 0
# makes it an exact identity (cos 0 = 1, sin 0 = 0).
_NO_PITCH = {"pitch": 0.0}


# Propagate movement of the wall nodes to the volume nodes by inverse-distance weighting
class IDWarp_jax(csdl.CustomExplicitOperation):
    """IDW mesh deformation via the ``idwarp-jax`` package's kd-tree warper.

    Maps the replicated ``(M_global, dim)`` wall-node motions
    (``evaluate(bdry_motions, shape_parameters)``, usually
    ``shape_design.FFDShapeParameterization.wall_displacement``) to rank-local volume
    node motions, in geometry-node order. Because the map is closed form rather
    than a PDE solve, this is a ``CustomExplicitOperation`` -- CSDL calls
    ``compute`` and ``compute_jacvec_product``.

    Everything that is a property of the mesh rather than of the warp lives on
    the ``Mesh`` (mesh_manager): the wall and anchor node sets
    and their global numbering (``Mesh.boundary_node_set``), the tagged facets
    and their orientation, applying/resetting node motions, mesh quality, and
    the output files. The wall numbering here is the mesh's wall node set's.

    The warp itself is ``idwarp_jax.build_volume_pts_func``, a kd-tree IDWarp
    that batches its forward and reverse passes internally, so its memory stays
    bounded without any chunking here. ``err_tol`` is its pruning tolerance,
    relative to the IDW denominator; ``err_tol=0`` disables pruning and gives
    the exact dense IDW sum. This class only turns the mesh's wall and anchor
    node sets into the package's surface and moves arrays between CSDL and it.

    MPI
    ---
    Volume points are mutually independent and each one needs the *whole*
    surface -- which every rank already has, since ``bdry_motions`` is a
    replicated ``(M_global, dim)`` array. So each rank builds its own warper
    over its own interior nodes against the global surface. The forward pass
    needs no collectives at all; the reverse pass needs exactly one
    ``Allreduce``, in ``compute_jacvec_product``, to sum every rank's
    contribution to the gradient with respect to the replicated wall motions.

    Those collectives run inside csdl's JAX callbacks, so ``ordered_callbacks``
    must stay True: otherwise XLA may run two of them concurrently (e.g. the
    drag and lift reverse passes) and cross-pair the Allreduces across ranks.

    Anchor nodes
    ------------
    IDW is a *normalized* weighted average, so unlike an elasticity warp it
    has no farfield decay -- it reproduces rigid modes everywhere. A rigid wall
    translation of 0.1 moves a point 300 units away by 0.1. The non-moving
    boundaries are therefore made part of the prescribed surface, held at zero
    displacement; ``anchor_meshtag_vals`` selects them. Together with the
    package's mirror about y = 0 this gives the usual mesh-deformation
    boundary conditions:

        farfield  u = 0            ->  anchor nodes, prescribed zero
        symmetry  u_normal = 0     ->  mirror about y = 0
        wall      u = prescribed   ->  prescribed from bdry_motions

    Symmetry
    --------
    The package always mirrors the surface about y = 0, and pins surface nodes
    within ``symmetry_tolerance`` of that plane to it. In 3D that is the
    model's symmetry plane, which windtunnel_model also hard-codes at
    y = 0. The pin is applied to the wall motions here too (``_project``), so
    the wall nodes move exactly as the warp assumed they do.

    2D
    --
    The package is intrinsically three-dimensional (cross products, 3x3
    Rodrigues matrices) and gives zero area to faces with fewer than three
    nodes -- and a 2D boundary facet is a two-node line. So each boundary edge
    is extruded into a quad (``extrusion_width``). The 2D plane is embedded as
    the **x-z** plane -- not x-y, which the y = 0 mirror would reflect about the
    chord line -- with the ribbon spanning ``y = -w/2 .. +w/2`` and the volume
    nodes at y = 0. Every ribbon normal then has n_y = 0 exactly, so each
    rotation is about y, so every surface node contributes exactly zero y
    displacement; and each volume node is its own mirror image, so the mirrored
    contributions duplicate the real ones and change nothing but the cost.

    Parameters
    ----------
    mesh_obj : mesh_manager.Mesh, with its boundary facets
        tagged. Its ``output_suffix`` (Mesh.configure_output) names this
        class's memory logs.
    wall_meshtag_val : tag of the moving (design) surface.
    anchor_meshtag_vals : tags of the boundaries held at zero displacement.
        Pass ``()`` to disable anchoring -- but read the note above first.
    extrusion_width : 2D only; width of the extruded boundary ribbon. ``None``
        derives it from the smallest boundary edge. See
        ``_resolve_extrusion_width`` for why the default is what it is -- the
        choice trades IDW distance fidelity against the conditioning of Newell's
        method, and neither limit is safe.
    LdefFact : IDWarp's dimensionless deformation length:
        ``Ldef = LdefFact * Ldef0``, with ``Ldef0`` the largest distance from
        the wall nodes' centroid to a wall node. The package takes ``Ldef0``
        over every prescribed node, anchors included -- the farfield radius --
        so the factor is rescaled before it is passed on.
    err_tol : kd-tree pruning tolerance, see above. Measured against
        ``err_tol=0`` on the 2D NACA0012 mesh: 1e-6 gives 1.2e-3 relative
        motion error at half the cost; 1e-10 prunes almost nothing (cost of
        the exact sum). On the 3D wing mesh 1e-6 keeps ~25% of the dense
        interactions, with a ~47 GiB host route cache summed over all ranks.
    **idwarp_kwargs : forwarded to ``build_volume_pts_func``, over these
        defaults:

        - ``normal_eps = warp_eps = 1e-30``. The package defaults (1e-6) are
          dimensional: ``normal_eps`` is added to |2A|^2 of every face, which
          is ~1e-14 for a 2D ribbon quad and ~1e-10 for a small 3D wall face,
          and ``warp_eps`` floors every IDW distance at 1e-3. Either would
          wreck the warp.
        - ``symmetry_tolerance``: 0 in 2D, where no ribbon node may be pinned
          to y = 0; 1e-6 in 3D, the tolerance the model tags symmetry facets
          with.
        - ``target_batch_interactions = 1_000_000`` (package: 6e6). Each rank
          holds one batch of volume-point x source interactions in flight,
          and the reverse pass's working arrays scale with it: on the 3D wing
          mesh at err_tol=1e-6 they add ~0.9 GiB/rank at 1e6 vs ~2.4 GiB/rank
          at 6e6, with identical results and no measurable slowdown.
        - ``progress = print_timings = False``; this class reports its own
          timings.

        Everything else (``zeroCornerRotations``, ``cornerAngle``,
        ``useRotations``, ``rotation_eps``, ``bucket_size``, ``route_cache``,
        ...) is the package's default.
    """
    ordered_callbacks = True    # MPI collectives in compute and compute_jacvec_product

    def __init__(
        self,
        mesh_obj,
        wall_meshtag_val: int = 2,
        anchor_meshtag_vals=(3, 4),
        extrusion_width: float = None,
        LdefFact: float = 1.0,
        err_tol: float = 1.0e-6,
        **idwarp_kwargs,
    ):
        super().__init__()

        self.mesh = mesh_obj
        comm = self.mesh.mesh.comm

        self.dimensions = self.mesh.mesh.geometry.dim
        self.wall_meshtag_val = wall_meshtag_val
        self.anchor_meshtag_vals = tuple(anchor_meshtag_vals or ())
        self._extrusion_width_arg = extrusion_width
        self.LdefFact = float(LdefFact)
        self.err_tol = float(err_tol)
        self.N_local = self.mesh.n_owned_nodes

        # Columns of the package's 3D arrays that hold the physical coordinates.
        # In 2D, column 1 is the extrusion direction (see the class docstring).
        self._cols = [0, 2] if self.dimensions == 2 else [0, 1, 2]

        self._idwarp_kwargs = dict(
            normal_eps=1.0e-30, warp_eps=1.0e-30,
            symmetry_tolerance=0.0 if self.dimensions == 2 else 1.0e-6,
            target_batch_interactions=1_000_000,
            progress=False, print_timings=False)
        self._idwarp_kwargs.update(idwarp_kwargs)

        # ---- the prescribed surface: the mesh's wall and anchor node sets ---
        # A node shared by the wall and an anchor boundary stays with the wall:
        # it has to move.
        self.wall = self.mesh.boundary_node_set((wall_meshtag_val,))
        self.anchors = self.mesh.boundary_node_set(self.anchor_meshtag_vals,
                                                   exclude=self.wall)
        self.M_global = self.wall.n_global
        prescribed = np.zeros(self.N_local, dtype=bool)
        prescribed[self.wall.local_idx] = True
        prescribed[self.anchors.local_idx] = True
        self._interior_local_idx = np.flatnonzero(~prescribed).astype(np.int64)
        self._n_interior = self._interior_local_idx.size
        # Surface node ids: the wall block first, so a wall node id equals its
        # wall global id, then the anchor block.
        self._n_surface_nodes = self.wall.n_global + self.anchors.n_global

        self._build_global_surface_topology()
        self._build_warper()

        PETSc.Sys.syncPrint(
            "[IDWarp_jax] rank {}: N_local={}, M_local={}, anchor_local={}, "
            "interior={}".format(
                comm.Get_rank(), self.N_local, self.wall.n_local,
                self.anchors.n_local, self._n_interior))
        PETSc.Sys.syncFlush()
        PETSc.Sys.Print(
            "[IDWarp_jax] M_global={}, anchor_global={}, n_surface_rows={}, "
            "n_faces={}, LdefFact={}, err_tol={:.1e}".format(
                self.M_global, self.anchors.n_global, self._n_surface_rows,
                len(self._mesh_dict["faces"]["type"]), self.LdefFact, self.err_tol))


    # ==================================================================
    # construction helpers
    # ==================================================================

    def _build_global_surface_topology(self):
        """Assemble the replicated prescribed surface as the package's mesh dict.

        ``self._mesh_dict`` holds ``points`` (the surface rows, ``self._Xs0``)
        and ``faces`` (every face with type 1, so wall and anchor faces are all
        prescribed). Every rank ends up with byte-identical arrays, because
        every rank has to evaluate the IDW sum against the whole surface.
        """
        comm = self.mesh.mesh.comm
        dim = self.dimensions

        node_coords = np.zeros((self._n_surface_nodes, 3), dtype=np.float64)
        node_coords[:self.M_global, self._cols] = self.wall.coords
        node_coords[self.M_global:, self._cols] = self.anchors.coords

        all_tags = (self.wall_meshtag_val,) + self.anchor_meshtag_vals
        local_faces, local_facet_idx = self.mesh.tagged_facets(
            all_tags, np.concatenate([self.wall.geom_gids, self.anchors.geom_gids]))

        if self.dimensions == 2:
            self._n_layers = 2
            self._extrusion_width = self._resolve_extrusion_width(node_coords, local_faces)
            half_w = 0.5 * self._extrusion_width
            # Layer-major row numbering: row = layer * n_nodes + node_id, so the
            # wall rows of layer 0 are [0, M_global) and _expand/_fold are plain
            # reshapes. The ribbon runs along y, out of the embedded x-z plane.
            Xs0 = np.vstack([node_coords, node_coords])
            Xs0[:self._n_surface_nodes, 1] = -half_w
            Xs0[self._n_surface_nodes:, 1] = +half_w
            if local_faces.size:
                a = local_faces[:, 0]
                b = local_faces[:, 1]
                conn = np.column_stack([
                    a, b,
                    b + self._n_surface_nodes,
                    a + self._n_surface_nodes,
                ])
            else:
                conn = np.zeros((0, 4), dtype=np.int64)
        else:
            self._n_layers = 1
            self._extrusion_width = None
            Xs0 = node_coords
            conn = local_faces      # quads already in cyclic order

        n_flipped = 0
        if conn.shape[0]:
            cell_mids = np.zeros((local_facet_idx.size, 3), dtype=np.float64)
            cell_mids[:, self._cols] = \
                self.mesh.facet_cell_midpoints(local_facet_idx)[:, :dim]
            conn, n_flipped = self.mesh.orient_facets(conn, Xs0, cell_mids)

        # Rank-order concatenation, so every rank sees the same face ordering.
        gathered = comm.allgather(np.ascontiguousarray(conn, dtype=np.int32))
        gathered = [g for g in gathered if g.size]
        if gathered:
            global_conn = np.concatenate(gathered, axis=0)
        else:
            raise RuntimeError(
                "IDWarp_jax: no tagged boundary facets found for tags {}.".format(
                    (self.wall_meshtag_val,) + self.anchor_meshtag_vals))

        self._n_surface_rows = Xs0.shape[0]
        self._Xs0 = np.ascontiguousarray(Xs0, dtype=np.float64)
        self._mesh_dict = {
            "points": self._Xs0,
            "faces": {
                "points": np.ascontiguousarray(global_conn, dtype=np.int32),
                "type": np.ones(global_conn.shape[0], dtype=np.int32),
            },
        }

        self._check_surface_topology(n_flipped)


    def _resolve_extrusion_width(self, node_coords, local_faces):
        """Pick the 2D ribbon width: a fraction of the shortest boundary edge.

        Two effects pull in opposite directions, and the factor below is where
        they balance.

        Smaller w is better for the IDW geometry: w enters the distance as
        sqrt(d_xy^2 + w^2/4), so a width comparable to the near-wall spacing
        would blunt exactly the weight concentration that makes the warp follow
        the surface. At w = 0.1 * h_min the nearest volume node sees its
        distance inflated by 0.125%, which is nothing.

        Larger w is better for conditioning. Newell's method in idwarp_jax's
        normal.py sums cross products of *absolute* coordinates, whose
        magnitudes are set by the farfield, and those terms have to cancel down to twice the face area --
        which for a ribbon quad is only edge_length * w. The relative error in
        the resulting normal therefore goes as 1/w. Measured on the 2D mesh, as
        the error in reproducing a rigid surface translation:

            w = 1e-5   ->  6.5e-08        w = 1e-2   ->  6.7e-11
            w = 1e-4   ->  6.7e-09        w = 1e-1   ->  6.7e-12
            w = 1e-3   ->  6.7e-10        w = 1e+0   ->  6.6e-13

        0.1 * h_min puts that error around 1e-9 for an O(1) displacement, four
        orders below the smallest cell, at a distance distortion of 0.1%.
        (3D is unaffected: real surface faces have areas many orders larger than
        an extruded ribbon's, so nothing cancels catastrophically.)

        The minimum is taken over every prescribed boundary edge, so in practice
        it is set by the wall, where the edges are finest.
        """
        if self._extrusion_width_arg is not None:
            return float(self._extrusion_width_arg)
        comm = self.mesh.mesh.comm
        if local_faces.size:
            p0 = node_coords[local_faces[:, 0]]
            p1 = node_coords[local_faces[:, 1]]
            local_min = float(np.linalg.norm(p1 - p0, axis=1).min())
        else:
            local_min = np.inf
        global_min = comm.allreduce(local_min, op=MPI.MIN)
        if not np.isfinite(global_min) or global_min <= 0.0:
            global_min = 1.0
        return 1.0e-1 * global_min

    def _check_surface_topology(self, n_flipped):
        """Fail loudly if the ranks assembled different surfaces.

        Identical on every rank, byte for byte: otherwise ranks evaluate
        different surfaces and the gradients they reduce are incoherent. A
        content hash rather than a sum, so a permutation cannot slip through.
        """
        comm = self.mesh.mesh.comm
        import hashlib
        faces = self._mesh_dict["faces"]["points"]
        digest = (hashlib.sha1(faces.tobytes()).hexdigest(),
                  hashlib.sha1(self._Xs0.tobytes()).hexdigest())
        digests = comm.allgather(digest)
        if any(d != digests[0] for d in digests):
            raise RuntimeError(
                "IDWarp_jax: the assembled prescribed surface differs between "
                "ranks. Per-rank (faces, Xs0) hashes: {}".format(digests))

        PETSc.Sys.Print(
            "[IDWarp_jax] surface: {} faces of {} nodes, {} rows, "
            "{} reversed for outward winding".format(
                faces.shape[0], faces.shape[1], self._n_surface_rows,
                comm.allreduce(n_flipped, op=MPI.SUM)))

    def _build_warper(self):
        """Build this rank's idwarp_jax warper over its own interior nodes.

        The volume points are the interior nodes only: wall motions are
        prescribed and anchors are held at zero, so neither needs warping. A
        rank that owns no interior node gets one dummy point, because the
        package cannot preprocess an empty volume.
        """
        msh = self.mesh.mesh
        comm = msh.comm
        t0 = perf_counter()

        n_vol = max(self._n_interior, 1)
        self._Xv0 = np.zeros((n_vol, 3), dtype=np.float64)
        if self._n_interior:
            self._Xv0[:, self._cols] = \
                msh.geometry.x[:self.N_local, :self.dimensions][self._interior_local_idx]
        else:
            self._Xv0[0] = self._Xs0.mean(axis=0)

        # The package takes Ldef0 over every prescribed row, anchors included;
        # rescale so that Ldef = LdefFact * Ldef0 over the wall rows, as in IDWarp.
        wall_rows = (np.arange(self._n_layers)[:, None] * self._n_surface_nodes
                     + np.arange(self.M_global)).reshape(-1)
        ldef0_wall = float(compute_idwarp_reference_length(self._Xs0[wall_rows]))
        ldef0_surface = float(compute_idwarp_reference_length(self._Xs0))
        ldef_fact = self.LdefFact * ldef0_wall / ldef0_surface

        # Pinned by the package to y = 0, so pinned here too (see _project).
        tol = self._idwarp_kwargs["symmetry_tolerance"]
        self._on_symmetry = np.abs(self._Xs0[:self.M_global, 1]) < tol

        self._warp, self._warp_vjp, aux = build_volume_pts_func(
            self._mesh_dict, self._Xv0, jax.devices("cpu")[0], return_aux=True,
            LdefFact=ldef_fact, err_tol=self.err_tol, **self._idwarp_kwargs)
        self.Ldef = float(aux["kdtree"]["ldef"])

        # The package numbers its surface by np.unique over the face nodes; our
        # row indexing only holds if that is every row, in order. An unreferenced
        # row would also carry zero area -- a silent hole in the IDW sum.
        if not np.array_equal(aux["wall_ids"], np.arange(self._n_surface_rows)):
            raise RuntimeError(
                "IDWarp_jax: {} of {} prescribed surface rows are not referenced "
                "by any face.".format(
                    self._n_surface_rows - len(aux["wall_ids"]), self._n_surface_rows))
        # Strictly positive area weights -- catches degenerate or bowtie faces
        # that survive the orientation pass.
        Ai = np.asarray(aux["kdtree"]["surface_area"])
        if not np.all(Ai > 0.0):
            raise RuntimeError(
                "IDWarp_jax: {} prescribed surface nodes have non-positive area "
                "weight (min {:.3e}); the surface faces are degenerate or wound "
                "inconsistently.".format(int((Ai <= 0.0).sum()), float(Ai.min())))

        # Compile every batch shape now, not inside csdl's jax.pure_callback:
        # compilation, with its thread spawning under a held GIL, is the risky
        # part of re-entering JAX from a callback.
        self._warp(self._Xs0, _NO_PITCH)
        self._warp_vjp(self._Xs0, _NO_PITCH, np.zeros_like(self._Xv0))

        routes_gib = aux["kdtree"]["routes"].nbytes / 2**30
        cache_gib = aux["kdtree"]["route_cache_nbytes"] / 2**30
        PETSc.Sys.Print(
            "[IDWarp_jax] warper: Ldef = {:.4e} (LdefFact {} x wall Ldef0 {:.4e}), "
            "Ai in [{:.3e}, {:.3e}], routes max {:.3f} GiB/rank + host cache max "
            "{:.3f} GiB/rank, setup incl. compilation {:.2f} s".format(
                aux["kdtree"]["ldef"], self.LdefFact, ldef0_wall,
                float(Ai.min()), float(Ai.max()),
                comm.allreduce(routes_gib, op=MPI.MAX),
                comm.allreduce(cache_gib, op=MPI.MAX),
                comm.allreduce(perf_counter() - t0, op=MPI.MAX)))

    # ==================================================================
    # array plumbing between the CSDL variables and the idwarp_jax arrays
    # ==================================================================

    def _project(self, wall_vals):
        """Zero the y component of wall nodes on the symmetry plane.

        Takes and returns a full (M_global, dim) array. The package pins those
        nodes to y = 0 before warping, so this is the wall motion the warp
        actually assumed. A diagonal projection, hence its own adjoint.
        Identity in 2D, where no node is pinned.
        """
        if not self._on_symmetry.any():
            return wall_vals
        wall_vals = np.array(wall_vals, dtype=np.float64)
        wall_vals[self._on_symmetry, 1] = 0.0
        return wall_vals

    def _expand(self, wall_vals):
        """(M_global, dim) wall values -> (n_surface_rows, 3), in every layer.

        The anchor rows stay zero.
        """
        out = np.zeros((self._n_layers, self._n_surface_nodes, 3), dtype=np.float64)
        out[:, :self.M_global][..., self._cols] = wall_vals[:, :self.dimensions]
        return out.reshape(-1, 3)

    def _fold(self, rows):
        """Adjoint of ``_expand``: (n_surface_rows, 3) -> (M_global, dim).

        The layers are summed because the forward writes the same wall motion
        into every layer. The anchor rows are dropped: they are sensitivities
        with respect to a constant.
        """
        rows = np.asarray(rows).reshape(self._n_layers, self._n_surface_nodes, 3)
        return np.ascontiguousarray(
            rows[:, :self.M_global].sum(axis=0)[:, self._cols], dtype=np.float64)

    def _assert_planar(self, motion_3d):
        """In 2D the out-of-plane (y) motion must vanish identically.

        It does so exactly, per surface node: both nodal normals are in-plane, so
        the rotation is about y, so the y component of
        M (x_v - Xs0_i) + Xs_i - x_v is (+-w/2) + (-+w/2) = 0. A nonzero value
        here means the ribbon construction is wrong, not that a tolerance is
        tight.
        """
        if self.dimensions != 2 or motion_3d.size == 0:
            return
        max_dy = float(np.abs(motion_3d[:, 1]).max())
        scale = max(1.0, float(np.abs(motion_3d[:, self._cols]).max()))
        if max_dy > 1.0e-9 * scale:
            raise RuntimeError(
                "IDWarp_jax: 2D warp produced an out-of-plane motion of {:.3e} "
                "(in-plane scale {:.3e}). The extruded ribbon surface is not "
                "mirror-symmetric about y = 0.".format(max_dy, scale))

    def _assemble_motions(self, wall_motions, interior_3d):
        """(N_local, dim) mesh-node motions in geometry-node order.

        Owned wall nodes take their rows of the replicated wall array, interior
        nodes the warp output; anchor nodes stay zero.
        """
        motions = np.zeros((self.N_local, self.dimensions), dtype=np.float64)
        motions[self.wall.local_idx] = \
            wall_motions[self.wall.global_idx, :self.dimensions]
        if self._n_interior:
            self._assert_planar(interior_3d)
            motions[self._interior_local_idx] = \
                interior_3d[:self._n_interior][:, self._cols]
        return motions


    # ==================================================================
    # CSDL interface
    # ==================================================================

    def evaluate(self, bdry_motions: "csdl.Variable", shape_parameters: "csdl.Variable"):
        """Register the operation.

        ``shape_parameters`` carries no derivative -- it is declared only to keep
        every rank's reverse chain attached to the design variables, including
        ranks that own no wall nodes.
        """
        self.name = "IDWarp_jax"

        self.declare_input("bdry_motions", bdry_motions)
        self.declare_input("shape_parameters", shape_parameters)

        mesh_node_motions = self.create_output(
            "mesh_node_motions", shape=(self.N_local, self.dimensions))

        self.declare_derivative_parameters(
            of="mesh_node_motions", wrt="shape_parameters", dependent=False)

        return mesh_node_motions

    def compute(self, input_vals, output_vals):
        """Forward warp. No collectives: every rank already has the whole surface."""
        t0 = perf_counter()
        self._log_memory("before compute")

        wall = self._project(input_vals["bdry_motions"])
        Xv = self._warp(self._Xs0 + self._expand(wall), _NO_PITCH)
        motions = self._assemble_motions(wall, Xv - self._Xv0)
        output_vals["mesh_node_motions"] = motions

        local_max = float(np.abs(motions).max()) if motions.size else 0.0
        PETSc.Sys.Print(
            "[IDWarp_jax] forward warp: max |motion| = {:.6e}, {:.2f} s".format(
                self.mesh.mesh.comm.allreduce(local_max, op=MPI.MAX), perf_counter() - t0))
        self._log_memory("after compute")

    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, mode):
        """Matrix-free VJP. Only reverse mode is supported.

        Each rank seeds only the
        rows it owns -- its own wall nodes for the identity block, its own
        interior nodes for the IDW block -- so the per-rank gradients are
        disjoint contributions to one (M_global, dim) array and summing them is
        exact.
        """
        t0 = perf_counter()
        self._log_memory("before compute_jacvec_product")
        Xs = self._Xs0 + self._expand(self._project(input_vals["bdry_motions"]))

        if mode == "rev":
            d_out = d_outputs.get(
                "mesh_node_motions",
                np.zeros((self.N_local, self.dimensions), dtype=np.float64))

            seed = np.zeros_like(self._Xv0)
            seed[:self._n_interior][:, self._cols] = d_out[self._interior_local_idx]
            d_Xs, _ = self._warp_vjp(Xs, _NO_PITCH, seed)

            # Identity block: the owned wall rows of the output are the wall
            # motions themselves.
            identity = np.zeros((self.M_global, self.dimensions), dtype=np.float64)
            identity[self.wall.global_idx] = d_out[self.wall.local_idx]
            grad = self._project(self._fold(d_Xs) + identity)

            # Reduce here, unconditionally, and with the buffer-mode collective.
            # Every rank reaches this with a real data dependence, which is what
            # keeps it from being scheduled out of step -- the reverse chain on a
            # rank owning no wall nodes would otherwise fold to a compile-time
            # constant and its collective could run at any point relative to the
            # others. See shape_parameterization.WallFFD's docstring for the
            # full account.
            # The lowercase (pickle) form is avoided because this array is
            # (M_global, dim) and that channel is shared by every lowercase
            # collective on the communicator.
            grad_combined = np.empty_like(grad)
            self.mesh.mesh.comm.Allreduce(grad, grad_combined, op=MPI.SUM)
            d_inputs["bdry_motions"] = grad_combined

            PETSc.Sys.Print(
                "[IDWarp_jax] reverse: |grad| = {:.6e}, {:.2f} s".format(
                    float(np.linalg.norm(grad_combined)), perf_counter() - t0))

        else:
            raise ValueError("Reverse mode required")

        self._log_memory("after compute_jacvec_product")

    # ==================================================================
    # diagnostics
    # ==================================================================

    def _log_memory(self, note):
        if self.mesh.output_suffix is None:
            return
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rank = MPI.COMM_WORLD.Get_rank()
        fname = 'memory_logs/r{}_{}.log'.format(rank, self.mesh.output_suffix)
        try:
            with open(fname, 'a') as f:
                f.write('    {}  {}  {}  {}\n'.format(
                    datetime.datetime.now(), os.getpid(), mem,
                    "IDWarp_jax, " + note))
        except OSError:
            pass

