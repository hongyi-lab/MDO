#!/usr/bin/env python3
"""Full three-condition STW hybrid task; no CFD or training is launched."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mdo_demo.stw_coupled import Coupled
from scipy.optimize import minimize


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def save(out,name,result,fields):
    d=out/name; d.mkdir(parents=True,exist_ok=True); write(d/'result.json',result)
    for cond,arr in fields.items():np.savez_compressed(d/f'{cond}.npz',**arr)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--stage',choices=['probe','optimize'],default='probe'); ap.add_argument('--mesh',default='L4')
    a=ap.parse_args(); out=a.output; out.mkdir(parents=True,exist_ok=False)
    with (a.project/'manifests/compute.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        load=os.getloadavg(); mem=dict(s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())
        free=int(mem['MemAvailable'].split()[0])/1024**2
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
        disk=os.statvfs(a.project).f_bavail*os.statvfs(a.project).f_frsize/1024**3
        if load[0]>22 or free<10 or disk<5 or int(gpu.split(',')[1].split()[0])>30:raise RuntimeError('Shared resources busy')
        import torch
        torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True)
        data=np.load(a.project/'code/stw_public_polar_v1_20260928/input/input.npz'); native=data['native_vertices_m']
        protocol=dict(method='hybrid ATsurf_L cruise + OAS compressible VLM maneuvers + stiffened-composite TACS',
          official_task='STW Case 2',mesh=a.mesh,low_speed_model_is_not_FM=True,
          Mach=[.77,.4577,.4577],altitude_m=[10400.,0.,0.],load_factors=[1.,2.5,-1.],
          objective='two-stage Breguet fuel burn',TSFC_kg_per_Ns=18.1e-6,gravity_in_exponents=True,
          safety='1.5 times limit loads with material/local-global-stiffener buckling KS <=1',
          structural_parameterization='3 independent sizing variables per official panel; fixed 0.15m pitch',
          reference_validation='L4 uniform 30kPa lower pressure +2.5g self-weight; outputs saved',
          training=False,new_CFD=False,independent_final_validation=False,load_start=load,RAM_GiB=free,gpu=gpu,disk_GiB=disk,
          source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'scripts/run_stw_case2.py',ROOT/'scripts/stw_tacs_worker.py',*[ROOT/'src/mdo_demo'/f for f in ['stw_task.py','stw_coupled.py']]]})
        write(out/'protocol.json',protocol)
        start=time.perf_counter(); c=Coupled(a.project,ROOT,out,native,ROOT/f'input/wingbox-{a.mesh}-Order2.bdf')
        try:
            d=c.worker.design.copy(); twist=[0.,0.]; thick=[1.,1.,1.]
            base,fields=c.analyze(twist,thick,d); save(out,'initial',base,fields)
            print('INITIAL',json.dumps({k:v for k,v in base.items() if k!='conditions'}),flush=True)
            if a.stage=='probe':return
            if not base['coupled_converged']:raise RuntimeError('Baseline coupling failed; not optimizing invalid points')
            # Alternate all-panel sizing with five-variable shape updates. Each
            # shape objective includes three converged load cases and mission trim.
            history=[]; result=base; best=None; eval_count=0
            for outer in range(4):
                sd=fields['cruise']['design']; nodes=fields['cruise']['nodes']
                inner,arr=c.worker.request(dict(nodes=nodes,design=sd,loads_0=fields['pullup']['loads'],loads_1=fields['pushdown']['loads'],
                      inertia_0=fields['pullup']['inertia_vector'],inertia_1=fields['pushdown']['inertia_vector']),command='optimize')
                d=arr['design']; write(out/f'inner_{outer:02d}.json',inner)
                result,fields=c.analyze(twist,thick,d); save(out,f'outer_{outer:02d}',result,fields)
                if not result['coupled_converged']:raise RuntimeError('Outer coupling failed; evidence retained')
                memo={}; candidates=[]
                def evaluate(x):
                    nonlocal eval_count
                    key=tuple(map(float,x))
                    if key in memo:return memo[key]
                    lo=np.array([-4.,-4.,.95,.95,.95]); hi=np.array([4.,4.,1.5,1.5,1.5])
                    violation=float(np.maximum(lo-x,0).sum()+np.maximum(x-hi,0).sum())
                    if violation>1e-9:
                        # COBYLA's bounds are constraints, not a promise that trial points stay inside.
                        memo[key]=(1e3+violation,np.full(3,-1.-violation));return memo[key]
                    eval_count+=1
                    try:
                        r,f=c.analyze(x[:2],x[2:],d)
                        score=r['mission']['fuel_burn_kg']/15000.
                        if r['coupled_converged']:
                            constraints=np.array([r['fuel_capacity_margin_kg']/15000.,
                                  1.-r['conditions']['pullup']['ultimate_failure'],
                                  1.-r['conditions']['pushdown']['ultimate_failure']])
                        else:constraints=np.full(3,-10.)
                        rec=dict(x=list(x),fuel=r['mission']['fuel_burn_kg'],constraints=constraints.tolist(),
                                 feasible=r['feasible_model_only'],coupled=r['coupled_converged'])
                        candidates.append((rec,r,f)); save(out,f'shape_{eval_count:04d}',r,f)
                    except (ValueError,RuntimeError) as exc:
                        score=1e3;constraints=np.full(3,-100.)
                        write(out/f'shape_{eval_count:04d}_failure.json',dict(x=list(x),error=str(exc)))
                    memo[key]=(score,constraints)
                    print('SHAPE',eval_count,list(x),score,constraints.tolist(),flush=True)
                    return memo[key]
                x0=np.r_[np.array(twist)/2.,thick]
                # Airfoil thickness factors are >=.95: spar height >=.75,
                # TE thickness >=.5, LE radius scales by factor^2 >=.9025.
                bounds=[(-2.,2.),(-2.,2.),(.95,1.5),(.95,1.5),(.95,1.5)]
                physical=lambda x:np.r_[2.*np.asarray(x[:2]),x[2:]]
                if outer==0:
                    for tw,th in [([-1.,-2.],[1.,1.,1.]),([-2.,-4.],[1.,1.,1.]),
                                  ([-1.,-3.],[1.15,1.1,1.]),([-2.,-3.],[1.25,1.1,1.])]:
                        evaluate(np.r_[tw,th])
                    seed=min([v[0] for v in candidates if v[0]['coupled']],
                        key=lambda v:(sum(min(z,0.)**2 for z in v['constraints']),v['fuel']))
                    x0=np.r_[np.array(seed['x'][:2])/2.,seed['x'][2:]]
                shape_opt=minimize(lambda x:evaluate(physical(x))[0],x0,method='COBYLA',bounds=bounds,
                    constraints={'type':'ineq','fun':lambda x:evaluate(physical(x))[1]},
                    options={'maxiter':18 if outer<2 else 12,'rhobeg':.15,'tol':.008,'catol':2e-4})
                # Never trust solver success alone. Select by independently checked constraints.
                feasible=[v for v in candidates if v[0]['feasible']]
                if feasible:
                    chosen=min(feasible,key=lambda v:v[0]['fuel'])
                elif any(v[0]['coupled'] for v in candidates):
                    chosen=min([v for v in candidates if v[0]['coupled']],key=lambda v:(sum(min(z,0.)**2 for z in v[0]['constraints']),v[0]['fuel']))
                else:raise RuntimeError('No finite fully evaluated shape candidates')
                rec,result,fields=chosen; twist=rec['x'][:2]; thick=rec['x'][2:]
                d=fields['cruise']['design']; save(out,f'outer_{outer:02d}_selected',result,fields)
                write(out/f'shape_optimizer_{outer:02d}.json',dict(success=bool(shape_opt.success),message=str(shape_opt.message),
                   x=physical(shape_opt.x).tolist(),selected=rec))
                history.append(dict(outer=outer,fuel=result['mission']['fuel_burn_kg'],feasible=result['feasible_model_only'],
                                    inner_success=inner['success'],coupled=result['coupled_converged'],twist=twist,thickness=thick))
                write(out/'history.json',history)
                if result['feasible_model_only'] and (best is None or result['mission']['fuel_burn_kg']<best['fuel']):
                    best=dict(fuel=result['mission']['fuel_burn_kg'],outer=outer); save(out,'best',result,fields)
                if outer>=3 and result['feasible_model_only'] and abs(history[-1]['fuel']/history[-2]['fuel']-1)<2e-4:break
            if best:
                f=np.load(out/'best/cruise.npz'); br=json.loads((out/'best/result.json').read_text())
                final,ff=c.analyze(br['twist'],br['thickness'],f['design'],tol=5e-5,maxiter=100)
                save(out,'final_tighter_reanalysis',final,ff)
                if not final['feasible_model_only']:best['tighter_reanalysis_failed']=True
            complete=best is not None and not best.get('tighter_reanalysis_failed',False)
            write(out/'summary.json',dict(status='complete_model_feasible' if complete else 'no_verified_feasible_design',best=best,history=history,
                  wall_seconds=time.perf_counter()-start,fm_calls=c.fm.calls,vlm_calls=c.vlm.calls,structural_requests=c.worker.count,
                  independent_validation=False,model_only=True))
        finally:c.close()

if __name__=='__main__':main()
