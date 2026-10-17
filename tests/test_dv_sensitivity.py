"""
utils/dv_sensitivity.py: total-derivative norms per design parameter, the probe
on the simulator, and their use as weights of the ROM snapshot distance
(ReducedOrderModel.set_distance_sensitivities).
"""
import numpy as np
import pytest

from dragonfly_sim.core.reduced_order_model import ReducedOrderModel
from dragonfly_sim.utils.dv_sensitivity import (total_derivative_norms, stack_total_derivatives,
                                                attach_total_derivative_probe)


class _Var:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class _FakeSim:
    """The part of a csdl_alpha simulator that the probe reads."""

    def __init__(self, result):
        self.result = result
        x, a = _Var('x', (2,)), _Var('alpha', (1,))
        self.dv_meta = {x: {'l_ind': 0, 'u_ind': 2}, a: {'l_ind': 2, 'u_ind': 3}}
        f, c = _Var('drag', (1,)), _Var('lift', (1,))
        self.o_meta = {f: {'l_ind': 0, 'u_ind': 1}}
        self.c_meta = {c: {'l_ind': 0, 'u_ind': 1}}

    def compute_optimization_derivatives(self):
        return self.result


def _result():
    return {'f': np.array([2.0]), 'c': np.array([0.5]),
            'df': np.array([[3.0, 0.0, 0.0]]), 'dc': np.array([[4.0, 1.0, 0.0]])}


@pytest.mark.unit_test
def test_column_norms_and_relative_norms():
    J = stack_total_derivatives(np.array([[3.0, 0.0]]), np.array([[4.0, 1.0]]))
    norms = total_derivative_norms(J, output_values=np.array([2.0, 0.5]))
    assert norms['column_norms'] == pytest.approx([5.0, 1.0])
    assert norms['column_norms_relative'] == pytest.approx([np.hypot(1.5, 8.0), 2.0])


@pytest.mark.unit_test
def test_param_scales_convert_to_parameter_units():
    norms = total_derivative_norms(np.array([[2.0, 3.0]]), param_scales=np.array([2.0, 1.0]))
    assert norms['column_norms'] == pytest.approx([1.0, 3.0])


@pytest.mark.unit_test
def test_probe_passes_norms_on_and_detaches():
    sim = _FakeSim(_result())
    seen = []
    detach = attach_total_derivative_probe(sim, printer=lambda s: None, on_norms=seen.append)
    assert sim.compute_optimization_derivatives() is sim.result
    assert len(seen) == 1 and seen[0]['column_norms'].shape == (3,)
    with pytest.raises(ValueError):
        attach_total_derivative_probe(sim, printer=lambda s: None)
    detach()
    sim.compute_optimization_derivatives()
    assert len(seen) == 1


@pytest.mark.unit_test
def test_probe_failure_does_not_break_the_gradient():
    sim = _FakeSim(_result())

    def boom(norms):
        raise RuntimeError("consumer failed")

    attach_total_derivative_probe(sim, printer=lambda s: None, on_norms=boom)
    assert sim.compute_optimization_derivatives() is sim.result


@pytest.mark.unit_test
def test_no_weights_without_information():
    """Default: no probe, no information, no weighting -- also in local_weighted mode."""
    rom = ReducedOrderModel(basis_mode='local_weighted')
    assert rom.distance_weighting == 'total_derivative'
    assert rom._distance_weights(3) is None
    rom.parameter_scales = np.array([1.0, 2.0, 3.0])
    assert rom._distance_weights(3) == pytest.approx([1.0, 2.0, 3.0])


@pytest.mark.unit_test
def test_weights_from_probe_are_mean_normalized_and_multiply_the_scales():
    sim = _FakeSim(_result())
    rom = ReducedOrderModel(basis_mode='local_weighted', distance_weighting='total_derivative_raw')
    attach_total_derivative_probe(sim, printer=lambda s: None, on_norms=rom.set_distance_sensitivities)
    sim.compute_optimization_derivatives()
    w = rom._distance_weights(3)
    assert w == pytest.approx(np.array([5.0, 1.0, 0.0]) / 2.0)
    rom.parameter_scales = np.array([1.0, 2.0, 1.0])
    assert rom._distance_weights(3) == pytest.approx(np.array([5.0, 2.0, 0.0]) / 2.0)


@pytest.mark.unit_test
def test_weighting_off_or_unfit_or_unusable_keeps_the_plain_distance():
    norms = {'column_norms': np.array([1.0, 3.0]), 'column_norms_relative': None}
    rom = ReducedOrderModel(distance_weighting=None)
    rom.set_distance_sensitivities(norms)
    assert rom._distance_weights(2) is None

    rom = ReducedOrderModel()
    rom.set_distance_sensitivities(norms)           # no relative norms: raw fallback
    assert rom._distance_weights(2) == pytest.approx([0.5, 1.5])
    assert rom._distance_weights(3) is None         # does not fit the parameter vector
    rom.set_distance_sensitivities({'column_norms': np.zeros(2), 'column_norms_relative': None})
    assert rom._distance_weights(2) == pytest.approx([0.5, 1.5])   # unusable: previous kept


@pytest.mark.unit_test
def test_invalid_distance_weighting_raises():
    with pytest.raises(ValueError):
        ReducedOrderModel(distance_weighting='bogus')
