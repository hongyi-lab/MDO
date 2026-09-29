#!/usr/bin/env python3
"""Bounded spanwise sizing starts and full Case 2 shape completion on the server."""
import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.optimize import minimize
from mdo_demo.stw_coupled import Coupled
from mdo_demo.stw_task import sizing_margins


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def save(path,r,fields):
    path.mkdir(parents=True,exist_ok=True);write(path/'result.json',r)
    for name,data in fields.items():np.savez_compressed(path/f'{name}.npz',**data)


def violation(r):
    if not r['coupled_converged']:return 1e3
    return max(0.,-r['fuel_capacity_margin_kg']/15000.)**2+sum(max(0.,r['conditions'][k]['ultimate_failure']-1.)**2 for k in ['pullup','pushdown'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();code=Path(__file__).resolve().parents[1]
    with (a.project/'manifests/compute.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        load=os.getloadavg();mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
        disk=os.statvfs(a.project).f_bavail*os.statvfs(a.project).f_frsize/1024**3
        if load[0]>22 or int(mem['MemAvailable'].split()[0])<10*1024**2 or int(gpu.split(',')[1].split()[0])>30 or disk<5:
            raise RuntimeError('Shared resources busy')
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        candidates=[]
        for path in a.previous.glob('eval_*/result.json'):
            r=json.loads(path.read_text())
            if r['coupled_converged'] and r['geometry_bounds_satisfied']:
                candidates.append((violation(r),r['mission']['fuel_burn_kg'],path,r))
        _,_,seedpath,seed=min(candidates,key=lambda x:(x[0],x[1]))
        seedfields={name:dict(np.load(seedpath.parent/f'{name}.npz')) for name in ['cruise','pullup','pushdown']}
        out=a.output;out.mkdir(parents=True,exist_ok=False)
        write(out/'protocol.json',dict(seed=str(seedpath),seed_sha256=hashlib.sha256(seedpath.read_bytes()).hexdigest(),
          method='FM cruise / compressible VLM maneuvers / TACS composite stiffened shell',
          objective='STW Case 2 mission fuel with all three coupled load cases',
          changes='spanwise structural starting profiles, then unchanged 8-variable shape parameterization',
          root_scales=[.9,1.,1.1,1.2],tip_scales=[.25,.5,.75],taper_powers=[.5,1.,2.],web_scales=[.7,1.],
          shape_budget=100,stop_rule='stop at first fully feasible candidate passing tighter reanalysis; no optimality claim',CFD=False,training=False,threads=1,GPU=gpu,load=load,disk_GiB=disk,
          source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),code/'scripts/stw_tacs_worker.py',code/'src/mdo_demo/stw_coupled.py',code/'src/mdo_demo/stw_task.py']}))
        native=np.load(a.project/'code/stw_public_polar_v1_20260928/input/input.npz')['native_vertices_m']
        started=time.perf_counter();c=Coupled(a.project,code,out,native,code/'input/wingbox-L4-Order2.bdf')
        try:
            base=seedfields['cruise']['design'];nodes=seedfields['cruise']['nodes'];panels=c.worker.panels
            sizing=[];history=[];final=None
            for count,(root,tip,power,web) in enumerate(itertools.product([.9,1.,1.1,1.2],[.25,.5,.75],[.5,1.,2.],[.7,1.]),1):
                d=base.copy()
                for panel in panels:
                    j=panel['dv_start'];eta=max(0.,min(1.,panel['centroid'][1]/14.))
                    scale=tip+(root-tip)*(1.-eta)**power
                    if 'SKIN' not in panel['name']:scale*=web
                    d[j]=max(.0006,d[j]*scale)
                    d[j+2]=max(.0006,d[j+1]/30.,d[j+2]*scale)
                    d[j+1]=max(.0254,d[j+1],5.*d[j+2])
                if min(sizing_margins(d,panels).values()) < -1e-8:continue
                rows=[]
                for name in ['pullup','pushdown']:
                    f=seedfields[name]
                    r,_=c.worker.request(dict(nodes=nodes,design=d,loads=f['loads'],inertia_vector=f['inertia_vector']),{'ultimate':True})
                    rows.append(r)
                record=dict(id=count,root=root,tip=tip,power=power,web=web,mass=rows[0]['mass'],failure=[r['ultimate_failure'] for r in rows])
                history.append(record)
                print('PROFILE',json.dumps(record),flush=True)
                if max(record['failure'])<=.97:sizing.append((record['mass'],d.copy(),record))
            write(out/'profile_trials.json',history)
            # Retain the old candidate; select from actual full coupled reanalyses.
            sized=[(seed,seedfields)]
            for j,(_,d,record) in enumerate(sorted(sizing,key=lambda x:x[0])[:6]):
                r,f=c.analyze(seed['twist'],seed['thickness'],d,camber_delta=seed['camber_delta'])
                save(out/f'profile_coupled_{j:02d}',r,f)
                if r['coupled_converged']:sized.append((r,f))
                if r['feasible_model_only']:
                    tight,tf=c.analyze(r['twist'],r['thickness'],d,camber_delta=r['camber_delta'],tol=5e-5,maxiter=100)
                    save(out/f'profile_tight_{j:02d}',tight,tf)
                    if tight['feasible_model_only']:
                        save(out/'final_tighter_reanalysis',tight,tf);final=tight;break
            if final:
                write(out/'summary.json',dict(status='complete_model_feasible',final=final,
                    solver_success=None,solver_message='Stopped at first qualified full task design, per user priority',
                    evaluations=0,profile_evaluations=len(history),wall_seconds=time.perf_counter()-started,
                    fm_calls=c.fm.calls,vlm_calls=c.vlm.calls,structural_requests=c.worker.count,
                    independent_validation=False,optimality_certified=False,full_task_model_only=True))
                return
            seed,seedfields=min(sized,key=lambda x:(violation(x[0]),x[0]['mission']['fuel_burn_kg']))
            design=seedfields['cruise']['design'];write(out/'sizing_selected.json',seed)
            memo={};full=[];count=0
            class Completed(Exception):pass
            def evaluate(x):
                nonlocal count,final
                x=np.asarray(x,float);key=x.tobytes()
                if key in memo:return memo[key]
                lo=np.array([-2.,-2.,.95,.95,.95,-2.,-2.,-2.]);hi=np.array([2.,2.,1.5,1.5,1.5,2.,2.,2.])
                v=float(np.maximum(lo-x,0).sum()+np.maximum(x-hi,0).sum())
                if v>1e-9:return 1e3+v,np.full(3,-1.-v)
                count+=1
                try:
                    r,f=c.analyze(x[:2]*2,x[2:5],design,camber_delta=x[5:]*.01)
                    cons=np.array([r['fuel_capacity_margin_kg']/15000.,1-r['conditions']['pullup']['ultimate_failure'],1-r['conditions']['pushdown']['ultimate_failure']]) if r['coupled_converged'] else np.full(3,-10.)
                    score=r['mission']['fuel_burn_kg']/15000.
                    full.append((r,f,x.copy()));save(out/f'eval_{count:04d}',r,f)
                    print('FULL',count,'fuel',r['mission']['fuel_burn_kg'],'constraints',cons.tolist(),'feasible',r['feasible_model_only'],flush=True)
                    if r['feasible_model_only']:
                        tight,tf=c.analyze(x[:2]*2,x[2:5],design,camber_delta=x[5:]*.01,tol=5e-5,maxiter=100)
                        save(out/f'tight_eval_{count:04d}',tight,tf)
                        if tight['feasible_model_only']:
                            save(out/'final_tighter_reanalysis',tight,tf);final=tight;raise Completed()
                except (ValueError,RuntimeError) as exc:
                    score=1e3;cons=np.full(3,-100.);write(out/f'eval_{count:04d}_error.json',dict(x=x.tolist(),error=str(exc)))
                memo[key]=(score,cons);return memo[key]
            x0=np.r_[np.array(seed['twist'])/2.,seed['thickness'],np.array(seed['camber_delta'])/.01]
            opt=None
            try:
                evaluate(x0)
                opt=minimize(lambda x:evaluate(x)[0],x0,method='COBYLA',bounds=[(-2,2)]*2+[(.95,1.5)]*3+[(-2,2)]*3,
                    constraints={'type':'ineq','fun':lambda x:evaluate(x)[1]},options={'maxiter':100,'rhobeg':.08,'tol':.003,'catol':1e-6})
            except Completed:pass
            valid=sorted([v for v in full if v[0]['feasible_model_only']],key=lambda v:v[0]['mission']['fuel_burn_kg'])
            for j,(r,f,x) in enumerate(valid[:5] if final is None else []):
                tight,tf=c.analyze(x[:2]*2,x[2:5],design,camber_delta=x[5:]*.01,tol=5e-5,maxiter=100)
                save(out/f'tight_candidate_{j:02d}',tight,tf)
                if tight['feasible_model_only']:
                    save(out/'final_tighter_reanalysis',tight,tf);final=tight;break
            write(out/'summary.json',dict(status='complete_model_feasible' if final else 'no_feasible_design',
                final=final,solver_success=bool(opt.success) if opt else None,
                solver_message=str(opt.message) if opt else 'Stopped at first qualified full task design, per user priority',evaluations=count,
                profile_evaluations=len(history),wall_seconds=time.perf_counter()-started,
                fm_calls=c.fm.calls,vlm_calls=c.vlm.calls,structural_requests=c.worker.count,
                independent_validation=False,full_task_model_only=True))
        finally:c.close()


if __name__=='__main__':main()
