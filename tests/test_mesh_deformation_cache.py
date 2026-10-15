"""
Mesh.apply_node_motions remembers the design vector it was built for: an equal
design leaves the mesh (and its listeners and quality check) alone, a different
one warps again from the baseline.
"""
import numpy as np
import pytest
from mpi4py import MPI

from dragonfly_sim.core.mesh_manager import Mesh
from cases import jittered_square


pytestmark = [pytest.mark.unit_test]


@pytest.fixture
def mesh_obj():
    mo = Mesh(jittered_square(6, MPI.COMM_WORLD), lambda pts: np.zeros(pts.shape[1], dtype=bool))
    mo.n_listener_calls = 0
    mo.n_quality_calls = 0
    quality = mo._update_quality

    def counted_quality():
        mo.n_quality_calls += 1
        quality()

    def listener():
        mo.n_listener_calls += 1

    mo._update_quality = counted_quality
    mo.add_geometry_listener(listener)
    return mo


def _motions(mo, scale):
    n, gdim = mo.n_owned_nodes, mo.mesh.geometry.dim
    # smooth in the position, so every rank moves a shared node identically
    x = mo.baseline_nodes[:n, :gdim]
    return scale * np.column_stack([np.sin(3.0 * x[:, 1]), np.cos(2.0 * x[:, 0])])


def _owned(mo):
    return mo.mesh.geometry.x[:mo.n_owned_nodes, :mo.mesh.geometry.dim].copy()


def test_equal_design_skips_the_warp(mesh_obj):
    mo = mesh_obj
    design = np.array([0.1, -0.2, 0.3])
    valid = mo.apply_node_motions(_motions(mo, 0.01), design=design)
    x_first = mo.mesh.geometry.x.copy()
    assert (mo.n_listener_calls, mo.n_quality_calls) == (1, 1)

    # same design (a copy, as the CSDL callbacks pass): nothing is redone
    assert mo.apply_node_motions(_motions(mo, 0.01), design=design.copy()) is valid
    assert (mo.n_listener_calls, mo.n_quality_calls) == (1, 1)
    assert np.array_equal(mo.mesh.geometry.x, x_first)
    assert np.array_equal(mo.deformation_design, design)


def test_changed_design_warps_from_the_baseline(mesh_obj):
    mo = mesh_obj
    mo.apply_node_motions(_motions(mo, 0.01), design=np.array([1.0]))
    mo.apply_node_motions(_motions(mo, 0.02), design=np.array([2.0]))
    assert (mo.n_listener_calls, mo.n_quality_calls) == (2, 2)
    n, gdim = mo.n_owned_nodes, mo.mesh.geometry.dim
    # baseline + new motions, not baseline + both
    assert np.allclose(_owned(mo), mo.baseline_nodes[:n, :gdim] + _motions(mo, 0.02), atol=1e-14)
    assert np.array_equal(mo.deformation_design, [2.0])


def test_no_design_always_warps(mesh_obj):
    mo = mesh_obj
    mo.apply_node_motions(_motions(mo, 0.01))
    mo.apply_node_motions(_motions(mo, 0.01))
    assert (mo.n_listener_calls, mo.n_quality_calls) == (2, 2)
    assert mo.deformation_design is None
    # an unkeyed warp never matches a later keyed call
    mo.apply_node_motions(_motions(mo, 0.01), design=np.array([1.0]))
    assert mo.n_quality_calls == 3


def test_reset_clears_the_key(mesh_obj):
    mo = mesh_obj
    design = np.array([1.0, 2.0])
    mo.apply_node_motions(_motions(mo, 0.01), design=design)
    mo.reset_nodes()
    assert mo.deformation_design is None
    assert np.array_equal(mo.mesh.geometry.x, mo.baseline_nodes)
    mo.apply_node_motions(_motions(mo, 0.01), design=design)
    assert mo.n_quality_calls == 2


def test_different_length_design_is_different(mesh_obj):
    mo = mesh_obj
    mo.apply_node_motions(_motions(mo, 0.01), design=np.array([1.0, 2.0]))
    mo.apply_node_motions(_motions(mo, 0.01), design=np.array([1.0, 2.0, 3.0]))
    assert mo.n_quality_calls == 2


def test_failed_warp_leaves_no_key(mesh_obj):
    mo = mesh_obj
    design = np.array([1.0])
    mo.apply_node_motions(_motions(mo, 0.01), design=design)
    with pytest.raises(Exception):
        mo.apply_node_motions(np.zeros((3, 2)), design=np.array([2.0]))
    assert mo.deformation_design is None


def test_validity_flag_survives_a_skipped_call(mesh_obj):
    mo = mesh_obj
    design = np.array([1.0])
    # (the unit-square quads have a negative Jacobian determinant by
    # orientation, so is_valid is False here whatever the motion)
    valid = mo.apply_node_motions(_motions(mo, 0.01), design=design)
    assert valid is False
    assert mo.apply_node_motions(_motions(mo, 0.01), design=design) is False
    assert mo.n_quality_calls == 1
