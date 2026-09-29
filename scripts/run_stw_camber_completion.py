#!/usr/bin/env python3
"""Finish Case 2 shape search with permitted camber variables; all three conditions.

Only starts after the previous frozen optimization finishes. No CFD/training.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import hashlib
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.optimize import minimize
from mdo_demo.stw_coupled import Coupled


def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def save(path,r,f):
    path.mkdir(parents=True,exist_ok=True);write(path/'result.json',r)
    for name,data in f.items():np.savez_compressed(path/f'{name}.npz',**data)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();code=Path(__file__).resolve().parents[1]
    with (a.project/'manifests/compute.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());load=os.getloadavg()
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True)
        if load[0]>22 or int(mem['MemAvailable'].split()[0])<10*1024**2 or int(gpu.strip())>30:raise RuntimeError('Shared resources busy')
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        # Select the best actual fully coupled candidate from the preceding run.
        rows=[]
        for p in a.previous.glob('shape_*/result.json'):
            r=json.loads(p.read_text())
            if r['coupled_converged'] and r['geometry_bounds_satisfied']:
                violation=max(0.,-r['fuel_capacity_margin_kg']/15000.)**2+sum(max(0.,r['conditions'][k]['ultimate_failure']-1.)**2 for k in ['pullup','pushdown'])
                rows.append((violation,r['mission']['fuel_burn_kg'],p,r))
        if not rows:raise RuntimeError('No qualified coupled previous candidate')
        _,_,seedpath,seed=min(rows,key=lambda row:(row[0],row[1]));f=np.load(seedpath.parent/'cruise.npz');design=f['design']
        out=a.output;out.mkdir(parents=True,exist_ok=False)
        write(out/'protocol.json',dict(seed=str(seedpath),seed_sha256=hashlib.sha256(seedpath.read_bytes()).hexdigest(),
          method='ATsurf_L cruise + OAS compressible VLM maneuvers + TACS stiffened composite shell',
          camber_formula='16*x_c^2*(1-x_c)^2 * chord * spanwise_interpolated_delta',camber_bounds=[-.02,.02],
          thickness_bounds=[.95,1.5],twist_bounds=[-4.,4.],new_CFD=False,training=False,threads=1,
          initial_sweep=19,optimizer_max_evaluations=40,independent_CFD_validation=False,
          source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),code/'src/mdo_demo/stw_task.py',code/'src/mdo_demo/stw_coupled.py',code/'scripts/stw_tacs_worker.py']}))
        native=np.load(a.project/'code/stw_public_polar_v1_20260928/input/input.npz')['native_vertices_m']
        start=time.perf_counter();c=Coupled(a.project,code,out,native,code/'input/wingbox-L4-Order2.bdf')
        candidates=[];memo={};count=0
        def evaluate(x):
            nonlocal count
            # x = [twist/2 (2), thickness (3), camber/.01 (3)]
            x=np.asarray(x,float);key=x.tobytes()
            if key in memo:return memo[key]
            lo=np.array([-2.,-2.,.95,.95,.95,-2.,-2.,-2.]);hi=np.array([2.,2.,1.5,1.5,1.5,2.,2.,2.])
            v=float(np.maximum(lo-x,0).sum()+np.maximum(x-hi,0).sum())
            if v>1e-9:return 1e3+v,np.full(3,-1.-v)
            count+=1
            try:
                r,f=c.analyze(2*x[:2],x[2:5],design,camber_delta=.01*x[5:])
                if r['coupled_converged']:
                    cons=np.array([r['fuel_capacity_margin_kg']/15000.,1-r['conditions']['pullup']['ultimate_failure'],1-r['conditions']['pushdown']['ultimate_failure']])
                else:cons=np.full(3,-10.)
                score=r['mission']['fuel_burn_kg']/15000.;rec=dict(x=x.tolist(),feasible=r['feasible_model_only'],coupled=r['coupled_converged'],fuel=r['mission']['fuel_burn_kg'],constraints=cons.tolist())
                candidates.append((rec,r,f));save(out/f'eval_{count:04d}',r,f)
                write(out/'evaluations.json',[row[0] for row in candidates]);print('CAMBER',count,rec,flush=True)
            except (RuntimeError,ValueError) as exc:
                score=1e3;cons=np.full(3,-100.);write(out/f'eval_{count:04d}_error.json',dict(x=x.tolist(),error=str(exc)))
            memo[key]=(score,cons);return memo[key]
        try:
            x0=np.r_[np.array(seed['twist'])/2.,seed['thickness'],[0.,0.,0.]]
            patterns=[[0.,0.,0.]]+[[s,s,s] for s in [-2.,-1.5,-1.,-.5,.5,1.]]
            patterns += [[s,t,0.] for s in [-2.,-1.,0.,1.] for t in [-1.,0.,1.]]
            for cm in patterns:evaluate(np.r_[x0[:5],cm])
            known=[v for v in candidates if v[0]['coupled']]
            best=min(known,key=lambda v:(sum(min(z,0.)**2 for z in v[0]['constraints']),v[0]['fuel']))
            x0=np.array(best[0]['x'])
            opt=minimize(lambda x:evaluate(x)[0],x0,method='COBYLA',bounds=[(-2,2)]*2+[(.95,1.5)]*3+[(-2,2)]*3,
              constraints={'type':'ineq','fun':lambda x:evaluate(x)[1]},options={'maxiter':40,'rhobeg':.15,'tol':.01,'catol':1e-5})
            valid=sorted([v for v in candidates if v[0]['feasible']],key=lambda v:v[0]['fuel'])
            final=None
            for candidate in valid[:3]:
                rec,r,f=candidate;x=np.array(rec['x'])
                tight,ff=c.analyze(2*x[:2],x[2:5],design,camber_delta=.01*x[5:],tol=5e-5,maxiter=100)
                save(out/'final_tighter_reanalysis',tight,ff)
                if tight['feasible_model_only']:final=tight;break
            write(out/'summary.json',dict(status='complete_model_feasible' if final else 'no_feasible_design',
                  solver_success=bool(opt.success),solver_message=str(opt.message),final=final,
                  evaluations=count,fm_calls=c.fm.calls,vlm_calls=c.vlm.calls,wall_seconds=time.perf_counter()-start,
                  independent_validation=False,full_task_model_only=True))
        finally:c.close()
if __name__=='__main__':main()
