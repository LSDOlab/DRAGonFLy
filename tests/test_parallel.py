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
import tempfile
import time

import pytest

_CHILD = "DRAGONFLY_PARALLEL_TEST_CHILD"
HERE = os.path.dirname(os.path.abspath(__file__))
# The two launches run side by side (conftest.py starts them when the session starts, so they also
# overlap the serial tests). Two launches rather than one keep each pytest process under MPICH's 2048
# communicators: a process that builds hundreds of meshes and matrices leaks them until its garbage
# is collected -- one launch of every module ran out ("Too many communicators") at the 270th test.
LAUNCHES = {
    "regression_and_models": ["test_euler_regression.py", "test_turbulence_models.py", "test_reconstruction.py",
                              "test_derivatives.py", "test_time_integration.py", "test_windtunnel_forward.py",
                              "test_pod.py", "test_reduced_order_model.py"],
    "caches_and_moments": ["test_mesh_deformation_cache.py", "test_incoming_state_converged.py",
                           "test_meshwarping_cache.py", "test_moments.py", "test_adjoint_reuse.py",
                           "test_boundary_tags.py"],
}
TIMEOUT = 900

_processes = {}
_markexpr = ""


pytestmark = [pytest.mark.verification, pytest.mark.unit_test]


def start_launches(markexpr=""):
    """Start every 3-rank launch (idempotent). Called by conftest.py at the start of a session.
    markexpr is pytest's -m expression, passed on so that the runs under MPI select the same tests."""
    global _markexpr
    _markexpr = markexpr
    if _processes or os.environ.get(_CHILD) == "1" or shutil.which("mpirun") is None:
        return
    env = dict(os.environ)
    env[_CHILD] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    for key in [k for k in env if k.startswith(("COV_CORE_", "COVERAGE_"))]:
        del env[key]
    for name, args in LAUNCHES.items():
        cmd = ["mpirun", "-n", "3", sys.executable, "-m", "pytest", "-x", "-q",
               "-p", "no:cacheprovider", "-p", "no:cov"] + (["-m", markexpr] if markexpr else []) + \
              [os.path.join(HERE, a) for a in args]
        # output to a file: a pipe nobody reads fills up and blocks the launch
        log = tempfile.TemporaryFile(mode="w+")
        _processes[name] = (subprocess.Popen(cmd, env=env, cwd=HERE, stdout=log, stderr=subprocess.STDOUT),
                            time.time(), log)


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun is not available")
@pytest.mark.skipif(os.environ.get(_CHILD) == "1", reason="already inside the MPI child")
@pytest.mark.parametrize("name", list(LAUNCHES))
def test_modules_under_three_ranks(name):
    start_launches()
    proc, started, log = _processes[name]
    try:
        proc.wait(timeout=max(TIMEOUT - (time.time() - started), 1))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail("the 3-rank run did not finish within {} s (a rank failing an assertion "
                    "leaves the others blocked in a collective)".format(TIMEOUT))
    log.seek(0)
    output = log.read()
    # exit code 5: the marker expression selects no test of this launch
    assert proc.returncode in ((0, 5) if _markexpr else (0,)), "3-rank run failed:\n{}".format(output[-8000:])
