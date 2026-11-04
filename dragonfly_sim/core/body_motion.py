"""
Prescribed rigid-body motion of the body on top of its shape deformation, for
time-dependent runs on a moving mesh.

The motion is tied to the FFD block of the shape parameterization
(core/shape_parameterization.py:WallFFD): it moves the wall nodes the block
embeds -- the nodes the shape deformation moves -- rigidly, by a translation
T(t) and a rotation R(theta(t)) about a centre c (the block's centroid unless
given):

    d(t) = d_s + [c + T(t) + R(theta(t)) (x_s - c) - x_s],    x_s = x_0 + d_s,

with x_0 the baseline wall nodes and d_s their shape (FFD) displacement. The
motion's displacement is added to the shape deformation's, evaluated on the
deformed wall, so the deformed body moves exactly rigidly. The volume mesh
follows through the mesh warper (core/meshwarping.py:IDWarp_jax), with the far
field held fixed, as for any shape change: the whole wall displacement d(t) is
warped once from the baseline mesh.

In 2D theta is the angle about +z (counter-clockwise; with the flow along +x
a nose-up pitch, which raises the angle of attack, is a negative theta); in 3D
it is a rotation vector (axis times angle). Time is the flow model's
nondimensional time, chords per free-stream speed.

MeshMotionUpdater moves the mesh in every time step of a
core/time_integration.py:TimeIntegrator and sets the model's mesh velocity
(CompressibleEulerModel.enable_mesh_motion), the BDF derivative of the node
positions, which is what the fluxes and the moving walls see.
"""
import time

import numpy as np


def rotation_matrix(theta, dim):
    """R(theta): the 2D rotation by the angle theta about +z, or the 3D one by
    the rotation vector theta (Rodrigues' formula)."""
    if dim == 2:
        c, s = np.cos(float(theta)), np.sin(float(theta))
        return np.array([[c, -s], [s, c]])
    theta = np.asarray(theta, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(theta)
    if angle == 0.0:
        return np.eye(3)
    k = theta / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


class PrescribedMotion:
    """
    A rigid-body motion of the wall, bound to an FFD block (bind).

    translation : t -> (dim,) displacement of the body, or None
    rotation    : t -> the rotation angle (2D) or rotation vector (3D), or None
    center      : the centre of the rotation, (dim,), at t = 0 (it moves with
                  the translation); None for the FFD block's centroid
    ramp        : None, or a start-up time T_r over which the motion is
                  faded in by r(t) = sin^2(pi t / (2 T_r)) (displacement and
                  angle alike), so it starts from rest instead of impulsively
    """

    def __init__(self, translation=None, rotation=None, center=None, ramp=None):
        self.translation = translation
        self.rotation = rotation
        self.center = None if center is None else np.asarray(center, dtype=np.float64)
        self.ramp = None if not ramp else float(ramp)
        self.wall_coords = None
        self.dim = None
        self._on_symmetry = None

    @classmethod
    def harmonic(cls, omega, plunge=None, pitch=None, pitch_phase=0.0, center=None, ramp=None):
        """
        Harmonic plunge and pitch at the angular frequency omega:

            T(t) = plunge sin(omega t),    theta(t) = pitch sin(omega t + pitch_phase),

        plunge a (dim,) amplitude vector, pitch an angle (2D, radians) or a
        rotation vector (3D).
        """
        plunge = None if plunge is None else np.asarray(plunge, dtype=np.float64)
        pitch = None if pitch is None else np.asarray(pitch, dtype=np.float64)
        translation = None if plunge is None else (lambda t: plunge * np.sin(omega * t))
        rotation = None if pitch is None else (lambda t: pitch * np.sin(omega * t + pitch_phase))
        return cls(translation, rotation, center=center, ramp=ramp)

    # ------------------------------------------------------------------
    def bind(self, ffd=None, corners=None, wall_coords=None, symmetry_tolerance=1e-6):
        """
        Tie the motion to an FFD block: `ffd` a WallFFD or an
        FFDShapeParameterization (after its setup), or the block's `corners`
        (shape_parameterization.ffd_corners_from_bounding_box /
        ffd_corners_from_corner_list) with the baseline `wall_coords` it
        embeds ((M_global, dim), replicated, in the mesh's wall node set
        order). The rotation centre defaults to the block's centroid. In 3D
        the wall nodes on the y = 0 symmetry plane must stay on it (the mesh
        warper pins them there), which the motion is checked against.
        Returns self.
        """
        if ffd is not None:
            ffd = getattr(ffd, "ffd", None) or ffd       # FFDShapeParameterization -> its WallFFD
            corners, wall_coords = ffd.corners, ffd.baseline_wall_coords
        if corners is None or wall_coords is None:
            raise ValueError("bind needs an FFD block, or its corners and the wall coordinates")
        self.wall_coords = np.ascontiguousarray(wall_coords, dtype=np.float64)
        self.dim = self.wall_coords.shape[1]
        corners = np.asarray(corners, dtype=np.float64).reshape(-1, self.dim)
        self.block_centroid = corners.mean(axis=0)
        if self.center is None:
            self.center = self.block_centroid.copy()
        if self.center.shape != (self.dim,):
            raise ValueError("the rotation centre has {} components, the wall is {}-D".format(
                self.center.size, self.dim))
        self._on_symmetry = (self.dim == 3
                             and bool(np.any(np.abs(self.wall_coords[:, 1]) < symmetry_tolerance)))
        return self

    def _require_bound(self):
        if self.wall_coords is None:
            raise RuntimeError("the motion is not bound to an FFD block yet (bind)")

    def _ramp(self, t):
        if self.ramp is None or t >= self.ramp:
            return 1.0
        return np.sin(0.5 * np.pi * max(t, 0.0) / self.ramp) ** 2

    def state(self, t):
        """(T(t), theta(t)) with the ramp applied."""
        self._require_bound()
        r = self._ramp(t)
        T = np.zeros(self.dim) if self.translation is None else \
            r * np.asarray(self.translation(t), dtype=np.float64).reshape(self.dim)
        n_rot = 1 if self.dim == 2 else 3
        theta = np.zeros(n_rot) if self.rotation is None else \
            r * np.asarray(self.rotation(t), dtype=np.float64).reshape(n_rot)
        if self._on_symmetry:
            if abs(T[1]) > 1e-12 or abs(theta[0]) > 1e-12 or abs(theta[2]) > 1e-12 \
                    or abs(self.center[1]) > 1e-12:
                raise ValueError("the motion moves wall nodes off the y = 0 symmetry plane: it needs "
                                 "T_y = 0, a rotation about y and a centre on the plane")
        return T, (theta[0] if self.dim == 2 else theta)

    def transform(self, t, x):
        """The body's rigid motion at t applied to points x ((n, dim) or (dim,))."""
        T, theta = self.state(t)
        R = rotation_matrix(theta, self.dim)
        return self.center + T + (np.asarray(x, dtype=np.float64) - self.center) @ R.T

    def wall_jacobian(self, t):
        """d d(t) / d d_s = R(theta(t)), the same for every wall node: the
        motion's linear map of the shape displacement (for an unsteady
        adjoint's chain through the warp)."""
        return rotation_matrix(self.state(t)[1], self.dim)

    def wall_displacement(self, t, shape_displacement=None):
        """d(t) (module docstring): the (M_global, dim) replicated displacement
        of the wall nodes, shape plus motion."""
        self._require_bound()
        d_s = 0.0 if shape_displacement is None else np.asarray(shape_displacement, dtype=np.float64)
        x_s = self.wall_coords + d_s
        return self.transform(t, x_s) - self.wall_coords


class MeshMotionUpdater:
    """
    Moves the mesh in every time step of the model's TimeIntegrator (attach;
    after any steady PTC phase, which refuses updaters) and sets the model's
    mesh velocity.

    model         a model with enable_mesh_motion() called
    node_motions  t -> the owned nodes' displacement from the baseline mesh,
                  (n_owned_nodes, gdim) in geometry-node order; for a body
                  motion see body() (the wall's displacement, warped)
    listeners     callables (t) run after every move, e.g. to move the moment
                  reference point with the body

    The positions are computed from the absolute time t^n + dt in prepare(),
    so a retried step (halved dt) simply places the mesh anew; the mesh
    velocity is the BDF derivative of the node positions with the step's
    coefficients, (a0 x^{n+1} + a1 x^n + a2 x^{n-1}) / dt (taken of the
    displacements from the baseline, which spares the round-off of the
    coordinates), which the moving walls and the fluxes see alike.
    """

    def __init__(self, model, node_motions, listeners=()):
        if model.grid_velocity is None:
            raise RuntimeError("the model has no mesh velocity: call enable_mesh_motion() before "
                               "defining its boundary conditions")
        self.model = model
        self.mesh_obj = model.mesh_obj
        self.node_motions = node_motions
        self.listeners = list(listeners)
        self.integrator = None
        self.time = None
        n, gdim = self.mesh_obj.n_owned_nodes, self.mesh_obj.mesh.geometry.dim
        self._x0 = self.mesh_obj.baseline_nodes[:n, :gdim].copy()
        # the displacements from the baseline at t^n, t^{n-1} and the trial t^{n+1}
        self._d_n = self._d_nm1 = self._d_trial = None
        self.seconds = 0.0          # spent moving the mesh

    @classmethod
    def body(cls, model, motion, warper, shape_displacement=None, listeners=()):
        """The updater of a bound PrescribedMotion: the wall displacement
        motion.wall_displacement(t, shape_displacement) (shape_displacement
        the FFD wall displacement d_s, (M_global, dim), or None for the
        baseline shape) warped to the volume by `warper`, the mesh's
        IDWarp_jax (built on the baseline mesh)."""
        def node_motions(t):
            return warper.warp(motion.wall_displacement(t, shape_displacement))
        updater = cls(model, node_motions, listeners)
        updater.motion = motion
        return updater

    def positions(self, t):
        """The owned nodes' positions at t, without moving the mesh."""
        return self._x0 + self.node_motions(t)

    def place(self, t):
        """Move the mesh to its position at t (the mesh velocity is left as it
        is); returns the positions. Collective."""
        return self._x0 + self._place(t)

    def _place(self, t):
        motions = self.node_motions(t)
        if not self.mesh_obj.apply_node_motions(motions, quiet=True):
            raise RuntimeError("the mesh motion inverts cells at t = {:.6g}".format(t))
        self.time = t
        for f in self.listeners:
            f(t)
        return motions

    def attach(self, integrator):
        """Register with `integrator` and place the mesh at its current time,
        with the history x^{n-1} at t - dt_prev (a restart). A start sets the
        mesh velocity to zero; a restart keeps the one already in the model's
        grid_velocity (restored from the checkpoint), which the forces and
        the residual of the restart state see. Collective."""
        self.integrator = integrator
        t = integrator.time
        self._d_n = self._place(t)
        if integrator.dt_prev is None:
            self._d_nm1 = self._d_n
            self.model.grid_velocity.x.array[:] = 0.0
        else:
            self._d_nm1 = self.node_motions(t - integrator.dt_prev)
        integrator.boundary_updaters.append(self)
        return self

    def prepare(self, order, ratio, dt):
        tic = time.perf_counter()
        self._d_trial = self._place(self.integrator.time + dt)
        a0, a1, a2 = (float(c.value) for c in self.integrator.term.a)
        w = (a0 * self._d_trial + a1 * self._d_n + a2 * self._d_nm1) / dt
        self.mesh_obj.nodal_function(w, out=self.model.grid_velocity)
        self.seconds += time.perf_counter() - tic

    def commit(self):
        self._d_nm1, self._d_n = self._d_n, self._d_trial
