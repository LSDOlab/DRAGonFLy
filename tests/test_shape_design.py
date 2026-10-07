"""Tests for shape_design: the design-variable layers on the wall FFD.

Serial, synthetic point clouds, no mesh: run with ``python -m pytest
test_shape_design.py``.
"""
import numpy as np
import pytest
from mpi4py import MPI
import csdl_alpha as csdl

from dragonfly_sim.core.shape_design import (FFDShapeParameterization, WingShape,
                                             ControlPointMotions, SectionalVariables)
from dragonfly_sim.core.shape_parameterization import WallFFD, SectionalShape
from dragonfly_sim.utils.ffd_dv_utils import ShapeDVSet, build_cp_motion_dv, build_sectional_dv
from test_shape_parameterization import half_wing, ellipse, WING_CORNERS

COMM = MPI.COMM_SELF
PLANFORM_BOUNDS = {'sweep': np.radians((10., 40.)), 'aspect_ratio': (6., 12.),
                   'root_chord': (4., 6.), 'taper_ratio': (0.2, 0.5)}


@pytest.fixture(autouse=True)
def recorder():
    rec = csdl.Recorder(inline=True)
    rec.start()
    yield rec
    rec.stop()


def swept_wing(n_span=15):
    """Swept, tapered half wing inside WING_CORNERS: LE 0.2 -> 7.5, TE 4.9 -> 8.9
    over a semi-span of 14, t/c 0.1."""
    return half_wing(span=14., le=(0.2, 7.5), te=(4.9, 8.9), t_over_c=0.1, n_span=n_span)


def wing_shape(X, tris, planform=PLANFORM_BOUNDS, sections=None, corners=WING_CORNERS,
               ffd_shape=(3, 5, 2), **kwargs):
    wing = WingShape(planform=planform, sections=sections, **kwargs)
    shape = FFDShapeParameterization(ffd_shape, [2, 2, 1], corners, layers=[wing])
    shape.setup(COMM, X, np.arange(len(X)), tris)
    shape.declare_design_variables()
    return shape, wing


def set_values(shape, values):
    for name, value in values.items():
        shape[name].value = np.atleast_1d(np.asarray(value, dtype=np.float64))


# ---- WingShape geometry ------------------------------------------------------

def test_wing_baseline_fit_recovers_the_trapezoid():
    X, tris = swept_wing()
    _, wing = wing_shape(X, tris)
    b = wing.baseline
    c_r, c_t, s = 4.7, 1.4, 14.
    ref_slope = (7.5 - 0.2 + 0.25 * ((8.9 - 7.5) - (4.9 - 0.2))) / s
    np.testing.assert_allclose(b['sweep'], np.arctan(ref_slope), rtol=1e-12)
    np.testing.assert_allclose(b['root_chord'], c_r, rtol=1e-12)
    np.testing.assert_allclose(b['taper_ratio'], c_t / c_r, rtol=1e-12)
    np.testing.assert_allclose(b['aspect_ratio'], 4 * s / (c_r + c_t), rtol=1e-12)
    np.testing.assert_allclose(b['area'], (c_r + c_t) * s, rtol=1e-12)


def test_wing_baseline_variables_are_the_identity():
    X, tris = swept_wing()
    shape, _ = wing_shape(X, tris, sections={'thickness': 0.1, 'camber': 0.01})
    np.testing.assert_allclose(shape.wall_displacement().value, 0., atol=1e-13)


def affine_planform_map(X, wing, sweep, aspect_ratio, root_chord, thickness=0.):
    """The exact image of the wall under a planform change at fixed taper."""
    b = wing.baseline
    taper = b['taper_ratio']
    s_new = aspect_ratio * root_chord * (1 + taper) / 4.
    r = root_chord / b['root_chord']
    y_new = X[:, 1] / b['semi_span'] * s_new
    x_ref0 = b['x_ref_root'] + X[:, 1] * np.tan(b['sweep'])
    z_c = np.polyval(b['z_chord_plane'], X[:, 1])
    return np.stack([b['x_ref_root'] + y_new * np.tan(sweep) + r * (X[:, 0] - x_ref0),
                     y_new,
                     z_c + r * (1. + thickness) * (X[:, 2] - z_c)], axis=1)


def test_planform_map_is_exact_with_fixed_taper():
    X, tris = swept_wing()
    shape, wing = wing_shape(X, tris, sections={'thickness': 0.1})
    values = dict(sweep=np.radians(31.), aspect_ratio=10.1, root_chord=5.3, thickness=0.07)
    set_values(shape, values)
    deformed = shape.wall_displacement().value + X
    np.testing.assert_allclose(deformed, affine_planform_map(X, wing, **values), rtol=0, atol=1e-11)

    # the measured planform of the deformed wall is the one the variables describe
    out = shape.outputs()['wing']
    geo = shape.geometry
    stations = geo.add_planform_stations(np.linspace(0., 13.5, 6))
    measured = geo.planform(shape.coefficients(), stations)
    s_new = 10.1 * 5.3 * (1 + wing.baseline['taper_ratio']) / 4.
    np.testing.assert_allclose(out['semi_span'].value, s_new, rtol=1e-14)
    np.testing.assert_allclose(measured['span'].value, 13.5 / 14. * s_new, rtol=1e-12)
    chords = 5.3 * (1. - (1. - wing.baseline['taper_ratio']) * np.linspace(0., 13.5, 6) / 14.)
    np.testing.assert_allclose(measured['chord'].value, chords, rtol=1e-12)


def test_taper_change_is_exact_at_the_root_and_close_elsewhere():
    X, tris = swept_wing()
    shape, wing = wing_shape(X, tris)
    set_values(shape, dict(taper_ratio=0.4))
    geo = shape.geometry
    y = np.linspace(0., 13.5, 10)
    stations = geo.add_planform_stations(y)
    measured = geo.planform(shape.coefficients(), stations)['chord'].value
    s_new = shape.outputs()['wing']['semi_span'].value
    target = 4.7 * (1. - 0.6 * (y / 14.))
    assert abs(measured[0] - target[0]) < 1e-10
    # B-spline blending of the section scalings between the FFD sections
    print("taper change 0.298 -> 0.4: max relative chord error {:.3e}".format(
        np.max(np.abs(measured - target) / target)))
    assert np.max(np.abs(measured - target) / target) < 0.03
    np.testing.assert_allclose(s_new, wing.baseline['aspect_ratio'] * 4.7 * 1.4 / 4., rtol=1e-14)


def test_thickness_and_camber_on_a_rectangular_wing():
    """On a rectangular, unswept wing every section is the same, so the
    sectional modes are exact: the thickness scales by 1 + tau and the camber
    line moves by kappa * c * 4 xi (1 - xi)."""
    X, tris = half_wing(span=14., t_over_c=0.12)
    shape, wing = wing_shape(X, tris, planform={}, sections={'thickness': 0.2, 'camber': 0.05},
                             corners=None)
    geo = shape.geometry
    handle = geo.add_thickness_stations([0.2, 0.5, 0.8], span_stations=[1., 7., 13.])
    set_values(shape, dict(thickness=np.full(5, 0.15), camber=np.full(5, 0.02)))
    np.testing.assert_allclose(geo.thickness_ratio(shape.coefficients(), handle).value, 1.15,
                               rtol=1e-10)
    dz = shape.wall_displacement().value[:, 2]
    xi = X[:, 0] / 5.
    camber = 0.02 * 5. * 4. * xi * (1. - xi)
    # thickness scales about the chord plane z = 0: dz = 0.15 z + camber
    np.testing.assert_allclose(dz, 0.15 * X[:, 2] + camber, rtol=0, atol=1e-12)


def test_wing_derivatives_match_central_differences():
    X, tris = swept_wing(n_span=8)
    shape, wing = wing_shape(X, tris, sections={'thickness': 0.2, 'camber': 0.05})
    point = dict(sweep=np.radians(29.), aspect_ratio=9.6, root_chord=4.9, taper_ratio=0.33,
                 thickness=np.array([0.02, -0.03, 0.05, 0.01, -0.04]),
                 camber=np.array([0.01, 0.005, -0.01, 0.02, 0.0]))
    set_values(shape, point)
    disp = shape.wall_displacement()
    out = shape.outputs()['wing']
    probe = disp[::7]
    ofs = [probe, out['area'], out['semi_span'], out['tip_chord']]
    wrts = [shape[name] for name in point]
    sim = csdl.experimental.PySimulator(csdl.get_current_recorder())
    an = sim.compute_totals(ofs, wrts)
    h = 1e-6
    for wrt in wrts:
        base = wrt.value.copy()
        for k in range(base.size):
            vals = []
            for sign in (1., -1.):
                pert = base.copy()
                pert[k] += sign * h
                sim[wrt] = pert
                sim.run()
                vals.append(np.concatenate([np.asarray(o.value).ravel() for o in ofs]))
            sim[wrt] = base
            fd = (vals[0] - vals[1]) / (2 * h)
            an_col = np.concatenate([np.asarray(an[o, wrt])[:, k] if np.asarray(an[o, wrt]).ndim > 1
                                     else np.atleast_1d(an[o, wrt])[k:k + 1] for o in ofs])
            np.testing.assert_allclose(an_col, fd, rtol=1e-6, atol=1e-8)


def test_wing_under_a_non_inline_recorder():
    """The drivers record with inline=False and change the design variables
    later. lsdo_geo freezes values when the graph is built; nothing of the
    wing layer may depend on the variables' values at that time."""
    X, tris = swept_wing(n_span=8)
    build_at = dict(sweep=np.radians(28.), aspect_ratio=9.5, root_chord=4.8, taper_ratio=0.35,
                    thickness=np.full(5, 0.05), camber=np.full(5, 0.01))
    run_at = dict(sweep=np.radians(22.), aspect_ratio=8.0, root_chord=5.4, taper_ratio=0.25,
                  thickness=np.linspace(-0.1, 0.1, 5), camber=np.linspace(0.02, -0.01, 5))
    sections = {'thickness': 0.2, 'camber': 0.05}

    def inline_displacement(values):
        rec = csdl.Recorder(inline=True)
        rec.start()
        shape, _ = wing_shape(X, tris, sections=sections)
        set_values(shape, values)
        disp = shape.wall_displacement().value
        rec.stop()
        return disp

    rec = csdl.Recorder(inline=False)
    rec.start()
    shape, _ = wing_shape(X, tris, sections=sections)
    set_values(shape, build_at)
    disp = shape.wall_displacement()
    rec.stop()
    sim = csdl.experimental.JaxSimulator(rec, additional_inputs=[shape[n] for n in run_at],
                                         additional_outputs=[disp], gpu=False)
    for name, value in run_at.items():
        sim[shape[name]] = np.atleast_1d(value)
    sim.run()
    np.testing.assert_allclose(np.array(sim[disp]), inline_displacement(run_at), rtol=0, atol=1e-12)


# ---- the generic layers -------------------------------------------------------

def test_layers_reproduce_the_manual_composition():
    X, segments = ellipse()
    rng = np.random.default_rng(4)
    camber = 0.003 * rng.uniform(-1, 1, 6)
    thickness = 0.003 * rng.uniform(-1, 1, 6)
    cp = 0.003 * rng.uniform(-1, 1, (6, 3, 1))

    shape = FFDShapeParameterization([6, 3], [2, 2], layers=[
        SectionalVariables(0, {'camber': 0.01, 'thickness': 0.01}),
        ControlPointMotions({1: 0.01})])
    shape.setup(COMM, X, np.arange(len(X)), segments)
    shape.declare_design_variables()
    set_values(shape, dict(camber=camber, thickness=thickness, cp_motions=cp))
    assert shape.dvs.names == ['camber', 'thickness', 'cp_motions']

    ffd = WallFFD(COMM, X, np.arange(len(X)), [6, 3], [2, 2])
    sec = SectionalShape(ffd, 0)
    sec.add('camber', csdl.Variable(value=camber))
    sec.add('thickness', csdl.Variable(value=thickness))
    C = ffd.apply_cp_motions(sec.apply(ffd.baseline_variable()), csdl.Variable(value=cp), [1])
    np.testing.assert_array_equal(shape.wall_displacement().value, ffd.wall_displacement(C).value)


def test_specification_checks():
    X, tris = swept_wing(n_span=6)
    with pytest.raises(ValueError, match="unknown planform"):
        WingShape(planform={'span': (10., 20.)})
    with pytest.raises(ValueError, match="unknown section"):
        WingShape(sections={'twist': 0.1})
    with pytest.raises(ValueError, match="bracket"):
        wing_shape(X, tris, planform={'aspect_ratio': (10., 12.)})
    with pytest.raises(ValueError, match="at least one layer"):
        FFDShapeParameterization([3, 5, 2])
    shape = FFDShapeParameterization([3, 5, 2], [2, 2, 1], WING_CORNERS, layers=[
        WingShape(planform={'sweep': PLANFORM_BOUNDS['sweep']}),
        SectionalVariables(0, {'thickness': 0.1})])
    shape.setup(COMM, X, np.arange(len(X)), tris)
    shape.declare_design_variables()
    with pytest.raises(ValueError, match="principal direction"):
        shape.coefficients()


def test_straight_edges_are_required():
    X, tris = swept_wing()
    bent = X.copy()
    bent[:, 0] += 0.3 * np.sin(np.pi * X[:, 1] / 14.)
    with pytest.raises(ValueError, match="straight"):
        wing_shape(bent, tris)


def test_build_planform_dv_scaler():
    from dragonfly_sim.utils.ffd_dv_utils import build_planform_dv
    dv = build_planform_dv(8.6, (7.5, 10.))
    np.testing.assert_array_equal(dv.value, [8.6])
    np.testing.assert_allclose(dv.scaler, [1. / 1.4])
