"""
Generate euler_reference_transient.npz: a BDF2 trajectory of the regression
channel (tests/cases.py:euler_channel_model, first-order Euler, seeded with
seed_mean_flow(amplitude=0.3) and released), in the former conservative
unknowns (rho, rho u, rho E) and the former time unit (chord / sqrt(p_inf /
rho_inf)): the state after every step of N_STEPS constant steps DT.

It was run ONCE on branch air_water_model @ b67683c, the last commit with
conservative unknowns: it needs that code (the seed writes conservative
values there) and cannot be regenerated from later commits.
test_time_integration.py checks that the primitive-variable model, with the
BDF term on U(V), follows the same trajectory: each step solves the same
discrete equations in other unknowns.

    python tests/data/make_euler_reference_transient.py tests/data/euler_reference_transient.npz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cases import euler_channel_model, seed_mean_flow, gather_by_original_cell  # noqa: E402
from dragonfly_sim.core.time_integration import StepSizeController  # noqa: E402

N_STEPS, DT = 8, 0.04

if __name__ == "__main__":
    model = euler_channel_model(n=8)
    seed_mean_flow(model, amplitude=0.3)
    model.report_positivity_limiter = False
    integrator = model.enable_time_stepping()
    model.set_up_solver()
    integrator.verbose = False
    integrator.start()
    control = StepSizeController(dt=DT, dt_start=DT)
    V = model.functionspaces["V"]
    bs = V.dofmap.index_map_bs
    states = []
    for _ in range(N_STEPS):
        rec = integrator.advance(control, order=2, newton_rtol=1e-13, newton_atol=1e-14, newton_max_it=30)
        assert rec.accepted
        states.append(gather_by_original_cell(
            V, model.u_vec.x.array[:V.dofmap.index_map.size_local * bs].reshape(-1, bs)))
    np.savez(sys.argv[1], states=np.array(states), dt=DT)
    print("saved", np.array(states).shape)
