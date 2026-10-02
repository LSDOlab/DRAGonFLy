"""Tests for shape_parameterization (lsdo_geo wall FFD) and the ffd_dv_utils sets.

Serial, synthetic point clouds, no mesh: run with ``python -m pytest
test_shape_parameterization.py``.
"""
import numpy as np
import pytest
from mpi4py import MPI
import csdl_alpha as csdl
import lsdo_function_spaces as lfs

from dragonfly_sim.core.shape_parameterization import (WallFFD, SectionalShape, WallGeometry,
                                                       ffd_corners_from_corner_list, validate_cp_coord_opt_idxs)
from dragonfly_sim.utils.ffd_dv_utils import (ShapeDVSet, build_cp_motion_dv, build_sectional_dv,
                                              spanwise_linear_bounds)

COMM = MPI.COMM_SELF
WING_CORNERS = [[(-0.1, -1e-8, 0.32), (5.1, -1e-8, -0.3)],
                [(7.4, 14.1, 0.32), (9.1, 14.1, -0.3)]]


@pytest.fixture(autouse=True)
def recorder():
    rec = csdl.Recorder(inline=True)
    rec.start()
    yield rec
    rec.stop()


def ellipse(n=400, a=0.5, b=0.06):
    th = np.linspace(0., 2 * np.pi, n, endpoint=False)
    X = np.stack([a + a * np.cos(th), b * np.sin(th)], axis=1)
    segments = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1)
    return X, segments


def half_wing(n_chord=60, n_span=8, span=14., le=(0., 0.), te=(5., 5.), t_over_c=0.12):
    """An elliptic-section half wing, open at y = 0 and closed by a flat tip cap.

    le/te give the leading/trailing edge x at the root and tip (linear in
    between); the default is a rectangular prism. Triangles, consistent winding.
    """
    th = np.linspace(0., 2 * np.pi, n_chord, endpoint=False)
    ys = np.linspace(0., span, n_span)
    X = []
    for y in ys:
        t = y / span
        x_le, x_te = (1 - t) * le[0] + t * le[1], (1 - t) * te[0] + t * te[1]
        c = x_te - x_le
        X += [[x_le + 0.5 * c * (1 + np.cos(a)), y, 0.5 * t_over_c * c * np.sin(a)] for a in th]
    X = np.array(X)
    quads = []
    for j in range(n_span - 1):
        for i in range(n_chord):
            a, b = j * n_chord + i, j * n_chord + (i + 1) % n_chord
            quads.append([a, b, b + n_chord, a + n_chord])
    tip_center = X.shape[0]
    X = np.vstack([X, [[0.5 * (le[1] + te[1]), span, 0.]]])
    # same winding as the side quads, whose top edge runs a+n -> b+n reversed
    tip = [[(n_span - 1) * n_chord + i, (n_span - 1) * n_chord + (i + 1) % n_chord, tip_center]
           for i in range(n_chord)]
    quads = np.array(quads)
    tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]], np.array(tip)])
    return X, tris


def swept_wing():
    """A swept, tapered half wing inside the WING_CORNERS block."""
    return half_wing(span=14., le=(0.2, 7.5), te=(4.9, 8.9), t_over_c=0.1)


def old_corner_list_lattice(shape, corner_list, margin=1e-4):
    """The pre-lsdo_geo lofted lattice (meshwarping_utils.
    construct_ffd_wrapping_block_from_corner_list), kept as the reference."""
    ndim = len(corner_list[0][0])
    mins = np.array([np.minimum(a, b) for a, b in corner_list], dtype=float)
    maxs = np.array([np.maximum(a, b) for a, b in corner_list], dtype=float)
    connecting = int(np.argmax(np.abs(0.5 * (mins[1] + maxs[1]) - 0.5 * (mins[0] + maxs[0]))))
    span_dirs = [d for d in range(ndim) if d != connecting]
    for s in range(2):
        for d in range(ndim):
            delta = maxs[s, d] - mins[s, d]
            if delta > 0:
                mins[s, d] -= margin * delta
                maxs[s, d] += margin * delta
    coords = [np.zeros(shape) for _ in range(ndim)]
    for j, t in enumerate(np.linspace(0., 1., shape[connecting])):
        lo = (1 - t) * mins[0] + t * mins[1]
        hi = (1 - t) * maxs[0] + t * maxs[1]
        grids = np.meshgrid(*[np.linspace(lo[d], hi[d], shape[d]) for d in span_dirs], indexing='ij')
        idx = [slice(None)] * ndim
        idx[connecting] = j
        for g, d in zip(grids, span_dirs):
            coords[d][tuple(idx)] = g
        coords[connecting][tuple(idx)] = lo[connecting]
    return np.stack(coords, axis=-1)


# ---- block construction and raw control-point motions ---------------------

def test_automatic_lattice_is_the_uniform_bounding_box_grid():
    X, _ = ellipse()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [10, 3], [2, 2])
    lo, hi = X.min(0), X.max(0)
    lo, hi = lo - 1e-4 * (hi - lo), hi + 1e-4 * (hi - lo)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 10), np.linspace(lo[1], hi[1], 3), indexing='ij')
    np.testing.assert_allclose(ffd.baseline_coefficients, np.stack([gx, gy], -1), rtol=0, atol=1e-15)


@pytest.mark.parametrize("shape", [[3, 5, 2], [10, 5, 2], [4, 5, 3]])
def test_corner_list_lattice_matches_the_old_lofted_block(shape):
    ref = old_corner_list_lattice(shape, WING_CORNERS)
    X, _ = swept_wing()
    ffd = WallFFD(COMM, X, np.arange(len(X)), shape, [2, 2, 1], ffd_block_corner_list=WING_CORNERS)
    np.testing.assert_allclose(ffd.baseline_coefficients, ref, rtol=0, atol=1e-13)


def test_wall_displacement_is_the_bspline_map_of_the_cp_motions():
    X, _ = ellipse()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [10, 3], [2, 2])
    m = 0.01 * np.random.default_rng(0).uniform(-1, 1, (10, 3, 1))
    C = ffd.apply_cp_motions(ffd.baseline_variable(), csdl.Variable(value=m), [1])
    disp = ffd.wall_displacement(C).value
    B = ffd.block.space.compute_basis_matrix(ffd.wall_parametric_coordinates)
    expected = np.zeros_like(X)
    expected[:, 1] = B @ m.reshape(-1)
    np.testing.assert_allclose(disp, expected, rtol=0, atol=1e-15)
    # the embedding reproduces the wall
    np.testing.assert_allclose(B @ ffd.baseline_coefficients.reshape(-1, 2), X, atol=1e-10)


def test_cp_motion_shape_and_direction_checks():
    X, _ = ellipse()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [5, 3], 2)
    with pytest.raises(ValueError, match="expected"):
        ffd.apply_cp_motions(ffd.baseline_variable(), csdl.Variable(value=np.zeros((5, 3, 2))), [1])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_cp_coord_opt_idxs([1, 0], 2)
    with pytest.raises(ValueError, match="lie in"):
        validate_cp_coord_opt_idxs([2], 2)


def test_probe_points_move_with_the_wall():
    X, _ = ellipse()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [10, 3], [2, 2])
    k = ffd.add_probe_points(X[::7])
    m = 0.01 * np.random.default_rng(1).uniform(-1, 1, (10, 3, 2))
    C = ffd.apply_cp_motions(ffd.baseline_variable(), csdl.Variable(value=m), [0, 1])
    np.testing.assert_allclose(ffd.probe_positions(C, k).value,
                               X[::7] + ffd.wall_displacement(C).value[::7], atol=1e-10)


# ---- sectional layer -------------------------------------------------------

def test_zero_sectional_variables_are_the_identity():
    X, _ = ellipse()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [10, 3], [2, 2])
    sec = SectionalShape(ffd, 0)
    sec.add('camber', csdl.Variable(value=np.zeros(10)))
    sec.add('thickness', csdl.Variable(value=np.zeros(4)))   # profile
    C = sec.apply(ffd.baseline_variable())
    np.testing.assert_array_equal(C.value, ffd.baseline_coefficients)


def test_camber_and_thickness_move_only_the_vertical_coordinate():
    X, _ = ellipse()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [10, 3], [2, 2])
    sec = SectionalShape(ffd, 0)
    sec.add('camber', csdl.Variable(value=np.full(10, 0.01)))
    sec.add('thickness', csdl.Variable(value=np.full(10, 0.02)))
    disp = ffd.wall_displacement(sec.apply(ffd.baseline_variable())).value
    assert np.abs(disp[:, 0]).max() < 1e-12
    # uniform camber shifts everything by 0.01; thickness adds a linear stretch
    height = ffd.corners[0, 1, 1] - ffd.corners[0, 0, 1]
    mid = 0.5 * (ffd.corners[0, 1, 1] + ffd.corners[0, 0, 1])
    np.testing.assert_allclose(disp[:, 1], 0.01 + 0.02 * (X[:, 1] - mid) / height, atol=1e-12)


def test_twist_keeps_the_root_on_the_symmetry_plane():
    X, _ = swept_wing()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [4, 5, 2], [2, 2, 1],
                  ffd_block_corner_list=WING_CORNERS)
    sec = SectionalShape(ffd, principal_dim=1)
    sec.add('twist', csdl.Variable(value=np.radians([1., 2., 3., 4., 5.])), pivot=[0.25, 0.5])
    sec.add('chord', csdl.Variable(value=np.full(5, 0.3)), pivot=[0., 0.5])
    disp = ffd.wall_displacement(sec.apply(ffd.baseline_variable())).value
    root = np.abs(X[:, 1]) < 1e-12
    assert np.abs(disp[root, 1]).max() < 1e-12
    assert np.abs(disp[root, 2]).max() > 1e-3   # but the root section did twist


def test_sectional_variable_size_is_checked():
    X, _ = ellipse()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [10, 3], [2, 2])
    with pytest.raises(ValueError, match="one per section"):
        SectionalShape(ffd, 0).add('camber', csdl.Variable(value=np.zeros(11)))
    with pytest.raises(ValueError, match="unknown sectional"):
        SectionalShape(ffd, 0).add('sweep', csdl.Variable(value=np.zeros(10)))
    with pytest.raises(ValueError, match="3-D block"):
        SectionalShape(ffd, 0).add('twist', csdl.Variable(value=np.zeros(10)))


# ---- geometric constraints -------------------------------------------------

def test_enclosed_area_and_thickness_of_an_ellipse():
    X, segments = ellipse()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [10, 3], [2, 2])
    # either winding gives the positive area
    for seg in (segments, segments[:, ::-1]):
        geo = WallGeometry(ffd, seg)
        polygon = 0.5 * np.abs(np.sum(X[:, 0] * np.roll(X[:, 1], -1) - np.roll(X[:, 0], -1) * X[:, 1]))
        assert geo.baseline_measure == pytest.approx(polygon, rel=1e-14)
    h = geo.add_thickness_stations([0.1, 0.5, 0.9])
    exact = 2 * 0.06 * np.sqrt(1 - (np.array([0.1, 0.5, 0.9]) - 0.5) ** 2 / 0.25)
    np.testing.assert_allclose(h['baseline'], exact, rtol=1e-3)
    C = ffd.baseline_variable()
    np.testing.assert_allclose(geo.thickness_ratio(C, h).value, 1., atol=1e-10)
    assert geo.enclosed_measure(ffd.wall_displacement(C)).value[0] == pytest.approx(polygon, rel=1e-12)


def test_half_wing_volume_thickness_and_planform():
    X, tris = half_wing()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [4, 5, 2], [2, 2, 1])
    geo = WallGeometry(ffd, tris)
    area_section = 0.5 * np.abs(np.sum(X[:60, 0] * np.roll(X[:60, 2], -1) - np.roll(X[:60, 0], -1) * X[:60, 2]))
    assert geo.baseline_measure == pytest.approx(area_section * 14., rel=1e-12)
    h = geo.add_thickness_stations([0.5], span_stations=[3., 10.])
    np.testing.assert_allclose(h['baseline'], 0.6, rtol=1e-3)
    p = geo.add_planform_stations([0., 7., 13.5])
    pf = geo.planform(ffd.baseline_variable(), p)
    np.testing.assert_allclose(pf['chord'].value, 5., rtol=1e-3)
    assert pf['span'].value[0] == pytest.approx(13.5)
    assert pf['area'].value[0] == pytest.approx(5. * 13.5, rel=1e-3)


def test_constraint_derivatives_match_finite_differences():
    X, segments = ellipse()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [6, 3], [2, 2])
    rng = np.random.default_rng(2)
    camber = csdl.Variable(value=0.003 * rng.uniform(-1, 1, 6), name='camber')
    thick = csdl.Variable(value=0.003 * rng.uniform(-1, 1, 3), name='thickness')
    cp = csdl.Variable(value=0.003 * rng.uniform(-1, 1, (6, 3, 2)), name='cp')
    sec = SectionalShape(ffd, 0)
    sec.add('camber', camber)
    sec.add('thickness', thick)
    C = ffd.apply_cp_motions(sec.apply(ffd.baseline_variable()), cp, [0, 1])
    geo = WallGeometry(ffd, segments)
    area = geo.enclosed_measure(ffd.wall_displacement(C))
    tr = geo.thickness_ratio(C, geo.add_thickness_stations([0.25, 0.6]))
    sim = csdl.experimental.PySimulator(csdl.get_current_recorder())
    an = sim.compute_totals([area, tr], [camber, thick, cp])
    # Central differences: the thickness is a norm, curved in the chordwise
    # motions, so csdl's forward differences carry an O(h/t) truncation error.
    h = 1e-6
    for wrt in (camber, thick, cp):
        base = wrt.value.copy()
        for k in range(base.size):
            vals = []
            for sign in (1., -1.):
                pert = base.copy().reshape(-1)
                pert[k] += sign * h
                sim[wrt] = pert.reshape(base.shape)
                sim.run()
                vals.append(np.concatenate([area.value.ravel(), tr.value.ravel()]))
            sim[wrt] = base
            fd_col = (vals[0] - vals[1]) / (2 * h)
            an_col = np.concatenate([an[area, wrt][:, k], an[tr, wrt][:, k]])
            np.testing.assert_allclose(an_col, fd_col, rtol=1e-6, atol=1e-9)


def test_wall_facet_checks():
    X, segments = ellipse()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [5, 3], 2)
    with pytest.raises(ValueError, match="index wall nodes"):
        WallGeometry(ffd, segments + 1)


# ---- design-variable sets ---------------------------------------------------

def test_shape_dv_set_layout():
    coeffs = np.random.default_rng(3).uniform(size=(4, 5, 2, 3))
    cp_dv = build_cp_motion_dv(coeffs, {2: spanwise_linear_bounds(0.1, 1., 0.5, 14.1)})
    tw_dv = build_sectional_dv(5, (np.radians([0., -2, -2, -2, -2]), np.radians([0., 2, 2, 2, 2])))
    dvs = ShapeDVSet()
    cp = dvs.add('cp_motions', cp_dv)
    assert dvs.flat_variable() is cp          # single variable passes through
    dvs.add('twist', tw_dv)
    assert dvs.size == 40 + 5 and dvs.shape == (45,)
    assert dvs.flat_variable().shape == (45,)
    with pytest.raises(ValueError, match="added twice"):
        dvs.add('twist', tw_dv)


def test_build_sectional_dv_bounds():
    dv = build_sectional_dv(3, 0.02)
    np.testing.assert_array_equal(dv.lower, -0.02)
    np.testing.assert_array_equal(dv.scaler, 50.)
    with pytest.raises(ValueError, match="non-positive"):
        build_sectional_dv(3, (0.01, 0.02))


def test_parameterization_solver_hits_a_planform_area_target():
    """lsdo_geo's ParameterizationSolver on sectional states: the chord
    stretches become implicit states driven so that the planform area equals a
    target, which is what the optimizer then sees as its design variable."""
    import lsdo_geo as lg
    X, _ = half_wing()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [4, 5, 2], [2, 2, 1])
    geo = WallGeometry(ffd, half_wing()[1])
    stations = geo.add_planform_stations([0., 3.5, 7., 10.5, 13.5])
    chord = csdl.Variable(value=np.zeros(5), name='chord_state')
    sec = SectionalShape(ffd, principal_dim=1)
    sec.add('chord', chord, pivot=[0., 0.5])
    C = sec.apply(ffd.baseline_variable())
    area = geo.planform(C, stations)['area']
    target = csdl.Variable(value=np.array([1.05 * 5. * 13.5]), name='area_target')
    solver = lg.ParameterizationSolver()
    solver.add_state(chord, cost=1.)
    gv = lg.GeometricVariables()
    gv.add_variable(computed_value=area, desired_value=target)
    solver.evaluate(gv)
    np.testing.assert_allclose(area.value, target.value, rtol=1e-8)
    # minimum-norm solution of a linear constraint: every section stretches
    assert np.all(chord.value > 0.)

    tip_chord = geo.planform(C, stations)['chord'][4]
    sim = csdl.experimental.PySimulator(csdl.get_current_recorder())
    an = sim.compute_totals([tip_chord], [target])[tip_chord, target]
    h = 1e-4
    vals = []
    for sign in (1., -1.):
        sim[target] = np.array([1.05 * 5. * 13.5 + sign * h])
        sim.run()
        vals.append(float(tip_chord.value[0]))
    np.testing.assert_allclose(an.ravel(), (vals[0] - vals[1]) / (2 * h), rtol=1e-5)


def test_sectional_layer_under_a_non_inline_recorder():
    """The drivers record with inline=False, where intermediate values are None
    until a simulator runs; lsdo_geo's stretch needs them at build time."""
    X, _ = swept_wing()
    ffd = WallFFD(COMM, X, np.arange(len(X)), [3, 3, 2], [2, 2, 1],
                  ffd_block_corner_list=WING_CORNERS)
    values = dict(twist=np.radians([0., .6, 1.2]), chord=np.array([0.02, 0.08]))

    def deformed(inline):
        rec = csdl.Recorder(inline=inline)
        rec.start()
        twist = csdl.Variable(value=values['twist'], name='twist')
        chord = csdl.Variable(value=values['chord'], name='chord')
        sec = SectionalShape(ffd, principal_dim=1)
        sec.add('twist', twist, pivot=[0.25, 0.5])
        sec.add('chord', chord, pivot=[0., 0.5])
        sec.add('thickness', csdl.Variable(value=np.full(3, 0.01)))
        disp = ffd.wall_displacement(sec.apply(ffd.baseline_variable()))
        rec.stop()
        if not inline:
            sim = csdl.experimental.JaxSimulator(rec, additional_inputs=[twist, chord],
                                                 additional_outputs=[disp], gpu=False)
            sim.run()
            return np.array(sim[disp])
        return disp.value

    np.testing.assert_allclose(deformed(False), deformed(True), rtol=0, atol=1e-13)
