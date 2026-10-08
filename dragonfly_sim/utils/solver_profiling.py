"""
Phase timing of the flow solves: where the wall time actually goes.

WHY THIS EXISTS
---------------
A projection-based ROM replaces the LINEAR SOLVE with a k x k one, but under
LSPG it still assembles the full-order residual and Jacobian at every Newton
step. So the assembly fraction of a FOM step is an Amdahl ceiling on what
projection alone can buy, and hyper-reduction is the only thing that attacks
it. Measuring that split turns "how much faster could this get" from a guess
into arithmetic:

    max speedup from projection alone  =  1 / assembly_fraction
    what hyper-reduction has to win    =  the assembly fraction itself

USE
---
Call sites wrap a region in ``with PROFILER.phase("group:name"):``; the group
(everything before the first ':') is the coarse bucket a breakdown is read in
(see default_group). The groups used in the package::

    assemble   FE residual and Jacobian assembly (the nonlinear problem, the
               condensed Green-Gauss gradient, the physical-residual check)
    solve      the Newton solve's linear algebra: the SNES/KSP solve with the
               nested assembly subtracted, and the reduced k x k solve
    rom        per-iteration ROM bookkeeping: basis construction, projection
               of the Jacobian and residual, expansion, line search
    limiter    the positivity-preserving step-length search
    setup      per-solve set-up: building the solver, the initial condition
    mesh       mesh deformation and reset (with the geometry listeners and
               quality metrics), and the IDWarp volume warp
    output     solution file writes
    post       force and coefficient evaluations for logging

Profiling is off by default and every phase() is then a no-op;
CompressibleEulerModel.enable_profiling() turns it on and profile_report()
prints the breakdown.

INCLUSIVE VS EXCLUSIVE
----------------------
Regions nest -- the Newton solve contains assembly, the ROM's line search
contains residual assemblies. A flat timer would double-count. Each frame
therefore subtracts the time its children consumed, so `self` times partition
the total exactly and can be summed, while `total` keeps the inclusive cost of
the enclosing operation. Read `self` for a cost breakdown; read `total` when
asking what one named operation costs end to end.

MPI ATTRIBUTION
---------------
Without a barrier, load imbalance in an unsynchronized phase is charged to
whichever COLLECTIVE call blocks next -- typically the KSP solve, which makes
assembly look cheap and the solve look expensive. That is the exact error
this module exists to avoid, so `barriers=True` is the default. It costs a
few percent of total runtime and makes the split honest. Turn it off only to
measure end-to-end wall time, where it perturbs nothing that matters. With
barriers on, every phase must be entered by every rank (all call sites in the
package are collective regions).

Timings are per-rank. report() reduces across ranks and prints max / mean,
because a phase's cost to the SOLVE is what the slowest rank spends in it.
"""

import time
from contextlib import contextmanager

import numpy as np


class PhaseProfiler:
    """Accumulates wall time per named phase. Disabled by default."""

    def __init__(self):
        self.enabled = False
        self.barriers = True
        self.comm = None
        # name -> [self_time, total_time, count]
        self._acc = {}
        # stack of [name, start_time, child_time]
        self._stack = []

    # ------------------------------------------------------------------
    def configure(self, comm=None, enabled=True, barriers=True):
        # Accept either an mpi4py comm or a petsc4py.PETSc.Comm: this module
        # needs Barrier/Allreduce/allgather, which only the former has, and a
        # PETSc comm would otherwise fail at the first phase edge rather than
        # at configure time.
        if comm is not None and not hasattr(comm, "allgather"):
            tompi4py = getattr(comm, "tompi4py", None)
            if tompi4py is None:
                raise TypeError(
                    "PhaseProfiler needs an mpi4py communicator (or a "
                    "petsc4py Comm convertible via tompi4py); got {!r}".format(
                        type(comm)))
            comm = tompi4py()
        self.comm = comm
        self.enabled = bool(enabled)
        self.barriers = bool(barriers)
        return self

    def reset(self):
        self._acc = {}
        self._stack = []

    # ------------------------------------------------------------------
    @contextmanager
    def phase(self, name):
        """
        Time a region. Nesting-aware: the enclosing region's `self` time
        excludes whatever is spent in here.

        A no-op fast path when disabled, so instrumented call sites cost
        nothing in production runs.
        """
        if not self.enabled:
            yield
            return
        if self.barriers and self.comm is not None and self.comm.size > 1:
            # Before the clock starts: absorb imbalance from the PREVIOUS
            # phase into that phase rather than into this one.
            self.comm.Barrier()
        frame = [name, time.perf_counter(), 0.0]
        self._stack.append(frame)
        try:
            yield
        finally:
            if self.barriers and self.comm is not None and self.comm.size > 1:
                self.comm.Barrier()
            elapsed = time.perf_counter() - frame[1]
            self._stack.pop()
            self_time = elapsed - frame[2]
            if self._stack:
                self._stack[-1][2] += elapsed
            rec = self._acc.setdefault(name, [0.0, 0.0, 0])
            rec[0] += self_time
            rec[1] += elapsed
            rec[2] += 1

    # ------------------------------------------------------------------
    def snapshot(self):
        """{name: (self, total, count)} as of now, this rank only."""
        return {k: tuple(v) for k, v in self._acc.items()}

    def report(self, title="", group_map=None, print_fn=None):
        """
        Print a max/mean-across-ranks breakdown, sorted by self time.
        Collective when the profiler has a communicator with more than one
        rank.

        `group_map` optionally maps a phase name to a coarse bucket (e.g.
        "assemble" / "solve"), which is what the ROM ceiling arithmetic
        actually needs; group subtotals are printed underneath.

        Returns {"names", "self_max", "total_max", "counts", "grand",
        "groups"}, or {} when nothing was recorded.
        """
        if print_fn is None:
            from petsc4py import PETSc
            print_fn = PETSc.Sys.Print

        names = sorted(self._acc)
        if self.comm is not None and self.comm.size > 1:
            # Union the key sets: a rank that never entered a phase (e.g. a
            # rank owning no boundary facets) would otherwise misalign the
            # reduction and silently pair different phases together.
            allnames = self.comm.allgather(names)
            names = sorted(set().union(*[set(a) for a in allnames]))
        if not names:
            print_fn("[profile] nothing recorded (profiling disabled?)")
            return {}
        self_v = np.array([self._acc.get(n, [0.0, 0.0, 0])[0] for n in names])
        tot_v = np.array([self._acc.get(n, [0.0, 0.0, 0])[1] for n in names])
        cnt_v = np.array([self._acc.get(n, [0.0, 0.0, 0])[2] for n in names])

        if self.comm is not None and self.comm.size > 1:
            from mpi4py import MPI
            s_max = np.zeros_like(self_v); s_sum = np.zeros_like(self_v)
            t_max = np.zeros_like(tot_v)
            self.comm.Allreduce(self_v, s_max, op=MPI.MAX)
            self.comm.Allreduce(self_v, s_sum, op=MPI.SUM)
            self.comm.Allreduce(tot_v, t_max, op=MPI.MAX)
            s_mean = s_sum / self.comm.size
            nranks = self.comm.size
        else:
            s_max, s_mean, t_max, nranks = self_v, self_v, tot_v, 1

        grand = s_max.sum()
        print_fn("")
        print_fn("=" * 78)
        print_fn("PHASE PROFILE{}  ({} rank{}, barriers={})".format(
            (": " + title) if title else "", nranks,
            "" if nranks == 1 else "s", self.barriers))
        print_fn("-" * 78)
        print_fn("{:<28} {:>10} {:>8} {:>10} {:>9} {:>8}".format(
            "phase", "self(max)", "share", "total(max)", "calls", "mean/rk"))
        order = np.argsort(-s_max)
        for i in order:
            if cnt_v[i] == 0 and s_max[i] == 0.0:
                continue
            print_fn("{:<28} {:>10.2f} {:>7.1f}% {:>10.2f} {:>9d} {:>8.2f}".format(
                names[i][:28], s_max[i], 100 * s_max[i] / max(grand, 1e-30),
                t_max[i], int(cnt_v[i]), s_mean[i]))
        print_fn("-" * 78)
        print_fn("{:<28} {:>10.2f} {:>7.1f}%".format("TOTAL (sum of self)", grand, 100.0))

        groups = {}
        if group_map:
            for i, n in enumerate(names):
                g = group_map(n)
                if g is None:
                    continue
                groups[g] = groups.get(g, 0.0) + s_max[i]
            if groups:
                print_fn("-" * 78)
                for g in sorted(groups, key=lambda k: -groups[k]):
                    print_fn("  {:<26} {:>10.2f} {:>7.1f}%".format(
                        g, groups[g], 100 * groups[g] / max(grand, 1e-30)))
                asm = groups.get("assemble", 0.0)
                if asm > 0 and grand > 0:
                    frac = asm / grand
                    print_fn("-" * 78)
                    print_fn("  assembly fraction        = {:.1%}".format(frac))
                    print_fn("  PER-STEP ceiling if the LINEAR SOLVE became free")
                    print_fn("                           = {:.2f}x  (Amdahl, 1/assembly_fraction)".format(
                        1.0 / max(frac, 1e-30)))
                    print_fn("  NOTE: per NEWTON STEP only. Total speedup is")
                    print_fn("        (FOM steps / ROM steps) x (per-step ratio), so a ROM")
                    print_fn("        that also converges in fewer steps can EXCEED this")
                    print_fn("        number.")
                    print_fn("  -> hyper-reduction is what attacks the assembly share; it is")
                    print_fn("     worth most once projection has removed the solve cost.")
        print_fn("=" * 78)
        return {"names": names, "self_max": s_max, "total_max": t_max,
                "counts": cnt_v, "grand": grand, "groups": groups}


def default_group(name):
    """Coarse bucket for a phase name: everything before the first ':'."""
    return name.split(":", 1)[0] if ":" in name else name


# Module-level singleton: import this, not a new instance, so every call site
# in every module accumulates into the same registry.
PROFILER = PhaseProfiler()
