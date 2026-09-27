#!/usr/bin/env python3
"""Host launcher for one checkpoint diagnostic; holds the shared project lock."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    path = args.manifest.resolve()
    state = json.loads(path.read_text())
    if state.get('status') != 'prepared':
        raise ValueError('Manifest is not a fresh prepared diagnostic')
    def save():
        path.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    with (project/'manifests/compute.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            state.update(status='resource_blocked', reason='Project compute.lock is held; no process was started')
            save()
            return 4
        # The lock covers this project only; preserve all other agents' jobs.
        memory_kib = int(next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:')))
        load = os.getloadavg()
        state['launch_resources'] = {'load_average':list(load), 'available_memory_kib':memory_kib,
                                     'logical_cpus':os.cpu_count(), 'mpi_ranks':8, 'threads_per_rank':1,
                                     'shared_machine':True}
        if memory_kib < 12*1024*1024 or load[0]+8 > (os.cpu_count() or 1):
            state.update(status='resource_blocked', reason='Insufficient shared CPU/memory margin; other jobs untouched')
            save()
            return 4
        mount = Path('/home/mdolabuser/mount')
        def inside(relative):
            checked = (project/relative).resolve()
            return str(mount/checked.relative_to(project))
        command = ['mpirun','-np','8','python','scripts/run_checkpoint_refinement.py',
                   '--source-result',inside(state['source_result']), '--request',inside(state['request']),
                   '--checkpoint',inside(state['checkpoint']), '--checkpoint-sha256',state['checkpoint_sha256'],
                   '--grid-audit',inside(state['grid_audit']), '--output',inside(state['output'])]
        seeded = state.get("kind") == "fixed_seed_initialization"
        if seeded:
            command[4] = "scripts/run_seed_initialization.py"
            command += ["--source-request", inside(state["source_request"]),
                        "--seed-cost-manifest", inside(state["seed_cost_manifest"])]
        command_text = 'source "$BASHRC_MDOLAB"; exec '+shlex.join(command)
        rootless = [sys.executable,str(project/'code/MDO/scripts/run_cfd_rootless.py'),
                    '--project',str(project),'--','/bin/bash','-c',command_text]
        start = time.perf_counter()
        state.update(status='running',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
                     numerical_policy={'preset':'robust_rans_ank_polish','max_cycles':1200,'l2_convergence':1e-10,'mpi_ranks':8},
                     command=command)
        save()
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        proc = subprocess.run(rootless, env=env, check=False)
        elapsed = time.perf_counter()-start
        state.update(status='completed' if proc.returncode == 0 else 'failed',exit_code=proc.returncode,
                     finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=elapsed,
                     source_plus_refinement_wall_seconds=state['source_stage_wall_seconds']+elapsed,
                     cost_scope='Source failed optimization stage plus complete refinement process; earlier independent failures and code development are additional disclosed development costs')
        if seeded:
            state.pop("source_plus_refinement_wall_seconds", None)
            state.update(seed_initialization_wall_seconds=elapsed,
                         shared_seed_preparation_seconds=state["source_stage_wall_seconds"],
                         cost_scope="This new-angle process measured independently. Immutable seed preparation is shared once; do not add its cost on every call. No MDO speedup reported.")
        save()
        (project/state['exit_file']).write_text(str(proc.returncode)+'\n')
        return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())
