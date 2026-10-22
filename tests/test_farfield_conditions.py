"""
The free-stream input and the far field built from it: the flat
{'rho', 'p', 'M', 'alpha'} dictionary, its validation, and the angle of
attack changing by large amounts (there is no inflow/outflow division to keep
valid, so nothing limits it).
"""
import os

import numpy as np
import pytest
from mpi4py import MPI

from dragonfly_sim.core.mesh_manager import Mesh
from dragonfly_sim.core.Euler_model import CompressibleEulerModel
from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model

from cases import FREESTREAM, euler_channel_model, jittered_square

from test_windtunnel_forward import bottom_wall, in_tmp_dir  # noqa: F401  (fixture)


def bare_model():
    mesh_obj = Mesh(jittered_square(4), lambda pts: np.zeros(pts.shape[1], dtype=bool))
    return CompressibleEulerModel(mesh_obj, poly_order=0)


def test_flat_dictionary_defines_the_free_stream():
    model = bare_model()
    model.define_farfield_conditions(FREESTREAM)
    ff = model.farfield_conditions
    # the solver's units: free-stream density and speed 1, so c = 1/M and
    # p_inf = 1/(gamma M^2); the dict's own units are kept for dimensional output
    M, g = FREESTREAM['M'], model.gamma
    assert ff['c'] == pytest.approx(1.0 / M)
    assert ff['p'] == pytest.approx(1.0 / (g * M**2))
    assert (ff['rho'], ff['M'], ff['alpha']) == (1.0, 0.5, FREESTREAM['alpha'])
    np.testing.assert_allclose(ff['flow_dir'].value, [np.cos(ff['alpha']), np.sin(ff['alpha'])])
    assert ff['u_ref'].value == pytest.approx(1.0)
    c = (g * FREESTREAM['p'] / FREESTREAM['rho']) ** 0.5
    assert model.material.dimensional == pytest.approx(
        {'rho': 1.0, 'p': 1.0, 'c': c, 'speed': M * c})


@pytest.mark.parametrize("bad", [
    {'inlet': {'rho': 1.0, 'p': 1.0, 'M': 0.5, 'alpha': 0.0}, 'outlet': {'p': 1.0}},   # the former format
    {'rho': 1.0, 'p': 1.0, 'M': 0.5},                                                  # alpha missing
    {'rho': 1.0, 'p': 1.0, 'M': 0.5, 'alpha': 0.0, 'outlet': {'p': 1.0}},              # extra key
])
def test_wrong_dictionary_is_rejected(bad):
    with pytest.raises(ValueError, match="flat dict"):
        bare_model().define_farfield_conditions(bad)


@pytest.mark.parametrize("M", [1.0, 1.5])
def test_supersonic_free_stream_is_not_implemented(M):
    with pytest.raises(NotImplementedError, match="subsonic"):
        bare_model().define_farfield_conditions(dict(FREESTREAM, M=M))


def test_large_alpha_changes_are_accepted():
    model = euler_channel_model(n=4)
    for deg in (12.0, -7.0, 30.0, 2.0):
        model.set_angle_of_attack(np.radians(deg))
        ff = model.farfield_conditions
        assert ff['alpha'] == pytest.approx(np.radians(deg))
        np.testing.assert_allclose(ff['flow_dir'].value, [np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        np.testing.assert_allclose(ff['inflow_angle'], ff['flow_dir'].value)


def test_windtunnel_tags_only_farfield_and_wall_and_solves_at_large_alpha(in_tmp_dir):  # noqa: F811
    wt = DG_windtunnel_model(jittered_square(8), dict(FREESTREAM, alpha=np.radians(2.0)),
                             mesh_inner_bdry_function=bottom_wall, poly_order=0,
                             filename_suffix="test_farfield")
    wt.euler_residual_conv_limit = 1e-10
    wt.set_up_sim()
    comm = wt.mesh.mesh.comm
    tags = set(sum(comm.allgather([int(t) for t in np.unique(wt.mesh.meshtags.values)]), []))
    assert tags == {wt.FARFIELD_TAG, wt.WALL_TAG}
    for deg in (2.0, 12.0, -6.0):
        _, converged, forces = wt.solve_forward(np.radians(deg))
        assert converged, deg
        assert np.isfinite(forces["c_l"])
