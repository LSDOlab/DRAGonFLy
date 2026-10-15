"""
incoming_state_converged: a state that already meets residual_conv_limit is
recognised from the residual form alone, so the caller can skip building the
solver (DG_windtunnel_model._run_fom_solve does).
"""
import pytest
from cases import euler_channel_model, seed_mean_flow


pytestmark = [pytest.mark.unit_test]


def test_converged_state_needs_no_solver():
    model = euler_channel_model(n=6)
    seed_mean_flow(model, amplitude=0.1)
    assert not model.incoming_state_converged()
    assert not model.solver_converged
    assert not model.solver_is_set_up

    model.set_up_solver()
    model.solve_system()
    assert model.solver_converged
    model.release_solver()
    assert not model.solver_is_set_up

    # converged: recognised without the solver, which stays unbuilt
    u = model.u_vec.x.array.copy()
    assert model.incoming_state_converged()
    assert model.solver_converged and not model.solver_is_set_up
    assert (model.u_vec.x.array == u).all()

    # perturbed: not converged, and the flag follows
    model.u_vec.x.array[:] *= 1.001
    model.u_vec.x.scatter_forward()
    assert not model.incoming_state_converged()
    assert not model.solver_converged
