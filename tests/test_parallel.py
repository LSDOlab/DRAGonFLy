"""
Re-run the RANS / time-integration / far-field / regression / POD modules under
mpirun, so a serial pytest invocation also covers the distributed path
(ghost cells in the Green-Gauss sums, the gathered wall geometry, the
partition-independent checkpoints, the shell dR/dx operator).
"""
import os
import shutil
import subprocess
import sys

import pytest

_CHILD = "DRAGONFLY_PARALLEL_TEST_CHILD"
HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ["test_euler_regression.py", "test_turbulence_models.py", "test_reconstruction.py",
           "test_derivatives.py", "test_time_integration.py", "test_windtunnel_forward.py",
           "test_pod.py", "test_reduced_order_model.py", "test_solver_profiling.py"]


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun is not available")
@pytest.mark.skipif(os.environ.get(_CHILD) == "1", reason="already inside the MPI child")
def test_modules_under_three_ranks():
    cmd = ["mpirun", "-n", "3", sys.executable, "-m", "pytest", "-x", "-q",
           "-p", "no:cacheprovider", "-p", "no:cov"] + [os.path.join(HERE, m) for m in MODULES]
    env = dict(os.environ)
    env[_CHILD] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    for key in [k for k in env if k.startswith(("COV_CORE_", "COVERAGE_"))]:
        del env[key]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900, cwd=HERE)
    except subprocess.TimeoutExpired:
        pytest.fail("the 3-rank run did not finish within 900 s (a rank failing an assertion "
                    "leaves the others blocked in a collective)")
    assert proc.returncode == 0, "3-rank run failed:\n{}\n{}".format(proc.stdout[-6000:], proc.stderr[-3000:])
