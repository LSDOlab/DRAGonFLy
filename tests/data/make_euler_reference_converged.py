"""
Generate euler_reference_converged.npz: the converged Newton solution of the
regression channel (tests/cases.py:euler_channel_model, seeded with
seed_mean_flow(amplitude=0.3)), with its residual history, in the former
conservative unknowns (rho, rho u, rho E).

It was run ONCE on branch air_water_model @ b67683c, the last commit with
conservative unknowns, before the switch to the primitive (p', u, T): it
needs that code (the seed writes conservative values there) and cannot be
regenerated from later commits. test_euler_regression.py compares the
primitive solver's converged solution against it.

    python tests/data/make_euler_reference_converged.py tests/data/euler_reference_converged.npz
"""
import sys, numpy as np
sys.path.insert(0, "/home/sebastiaan/Repositories/DRAGonFLy/tests")
from cases import euler_channel_model, seed_mean_flow, gather_by_original_cell
model = euler_channel_model(n=8)
V = model.functionspaces["V"]
seed_mean_flow(model, amplitude=0.3)
model.report_positivity_limiter = False
model.fom_newton_inner_max_it = 1
model.max_newton_iterations = 1
model.residual_conv_limit = 1e-300
model.set_up_solver()
hist = []
for _ in range(12):
    hist.append(model.solve_system().norm())
print(hist)
bs = V.dofmap.index_map_bs
state = gather_by_original_cell(V, model.u_vec.x.array[:V.dofmap.index_map.size_local * bs].reshape(-1, bs))
np.savez(sys.argv[1], converged_state=state, converged_history=np.array(hist))
