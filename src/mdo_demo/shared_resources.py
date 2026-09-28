"""Observe shared resources without changing or terminating any process."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone


def snapshot(project, *, need_gpu=False, ranks=8):
    if ranks not in (1, 8):
        raise ValueError('Only one CPU task or the fixed eight-rank CFD allocation is supported')
    cpus=os.cpu_count() or 1
    load=list(os.getloadavg())
    memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(memory['MemAvailable'].split()[0])/1024**2
    disk=shutil.disk_usage(project).free/1024**3
    gpu=None
    if need_gpu:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],text=True,timeout=15).strip().splitlines()
        if len(raw)!=1:
            raise ValueError('This frozen experiment expects one GPU')
        free,util=map(float,raw[0].split(','))
        gpu={'free_mib':free,'utilization_percent':util}
    ready=(cpus>=ranks and max(load[:2])<=cpus-ranks and available>=16 and disk>=20
           and (gpu is None or (gpu['free_mib']>=12000 and gpu['utilization_percent']<=30)))
    return {'utc':datetime.now(timezone.utc).isoformat(),'logical_cpus':cpus,
            'load_average':load,'available_ram_gib':available,'free_disk_gib':disk,
            'gpu':gpu,'requested_cpu_ranks':ranks,'ready':ready,
            'scope':'Shared load snapshot, not an exclusive CPU reservation; no process is modified.'}


def wait_for_resources(project, log_path, *, need_gpu=False, ranks=8):
    """Called in the background queue, before new compute, with its own lock.

    Waiting is recorded in the total clock. It does not consume a solve/retry
    budget. No global process locks, affinity changes or termination are used.
    """
    started=time.perf_counter()
    log_path=Path(log_path)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8') as stream:
        while True:
            state=snapshot(project,need_gpu=need_gpu,ranks=ranks)
            state['wait_wall_seconds']=time.perf_counter()-started
            stream.write(json.dumps(state,allow_nan=False)+'\n');stream.flush()
            if state['ready']:
                return state['wait_wall_seconds']
            time.sleep(30)
