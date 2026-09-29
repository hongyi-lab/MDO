#!/usr/bin/env python3
"""Persistent real TACS worker on official STW BDF; JSON file IPC, no CFD."""
import argparse
import json
from pathlib import Path
import sys
import os
import time
import traceback
import numpy as np
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/stw_numba_cache')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mdo_demo.stw_task import read_stw_bdf, G
from scipy.optimize import minimize, linprog
from mpi4py import MPI
from tacs import pyTACS, TACS, elements, constitutive, functions


class Structure:
    def __init__(self,bdf):
        self.mesh=read_stw_bdf(bdf); self.cons=[]; self.panels=[]
        self.fea=pyTACS(str(bdf),comm=MPI.COMM_SELF,options={'printTiming':False})
        props=constitutive.MaterialProperties(rho=1550.,E1=117.7e9,E2=9.7e9,nu12=.35,
             G12=4.8e9,G13=4.8e9,G23=4.8e9,Xt=1648e6,Xc=1034e6,Yt=64e6,Yc=228e6,S12=71e6)
        ply=constitutive.OrthotropicPly(.1,props)
        def callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kw):
            pid=kw['propID']; name=self.mesh['labels'][pid]
            inds=np.unique(self.mesh['conn'][self.mesh['prop']==pid]); x=self.mesh['nodes'][inds]
            skin='SKIN' in name
            rear_pids=[pid for pid,label in self.mesh['labels'].items() if 'SPAR.01/' in label]
            rear_ids=np.unique(self.mesh['conn'][np.isin(self.mesh['prop'],rear_pids)])
            rear=self.mesh['nodes'][rear_ids]; rear=rear[rear[:,1]>1.51]
            slope=np.polyfit(rear[:,1],rear[:,0],1)[0]
            axis=np.array([slope,1.,0.]) if skin else np.array([0.,0.,1.])
            if skin and x[:,1].max()<=1.501: axis=np.array([0.,1.,0.])
            axis/=np.linalg.norm(axis)
            length=float(np.ptp(x[:,1])/axis[1] if skin else np.ptp(x[:,2]))
            con=constitutive.BladeStiffenedShellConstitutive(panelPly=ply,stiffenerPly=ply,
                panelLength=length,stiffenerPitch=.15,panelThick=.0065,
                panelPlyAngles=np.deg2rad([0.,-45.,45.,90.]),
                panelPlyFracs=np.array([.4441,.222,.222,.1119] if skin else [.10,.35,.35,.20]),
                stiffenerHeight=.05,stiffenerThick=.006,
                stiffenerPlyAngles=np.deg2rad([0.,-45.,45.,90.]),
                stiffenerPlyFracs=np.array([.4441,.222,.222,.1119]),
                panelLengthNum=dvNum+3,stiffenerPitchNum=-1,panelThickNum=dvNum,
                stiffenerHeightNum=dvNum+1,stiffenerThickNum=dvNum+2,flangeFraction=1.)
            con.setPanelThicknessBounds(.0006,.06); con.setStiffenerHeightBounds(.0254,.149)
            con.setStiffenerThicknessBounds(.0006,.025)
            con.setFailureModes(includePanelMaterialFailure=True,includeStiffenerMaterialFailure=True,
                includeLocalBuckling=True,includeGlobalBuckling=True,includeStiffenerColumnBuckling=True,
                includeStiffenerCrippling=True)
            self.cons.append(con)
            self.panels.append(dict(comp=compID,prop=pid,name=name,dv_start=dvNum,length=length,
                         centroid=x.mean(axis=0).tolist(),axis=axis.tolist()))
            return elements.Quad4Shell(elements.ShellRefAxisTransform(axis),con),[100.,20.,100.,1.]
        self.fea.initialize(callback)
        self.nodes=self.fea.getOrigNodes().reshape(-1,3).copy()
        self.base_dv=self.fea.getOrigDesignVars().copy()
        self.problem=self.fea.createStaticProblem('stw',options={'printTiming':False,'printLevel':0,
                          'L2Convergence':1e-10,'L2ConvergenceRel':1e-10})
        self.problem.addFunction('mass',functions.StructuralMass)
        self.problem.addFunction('failure',functions.KSFailure,ksWeight=80.,ftype='discrete')
        self.problem.addFunction('compliance',functions.Compliance)
        # Ordering map to compare with and communicate in the official Nastran node order.
        self.to_tacs=np.asarray(self.fea.meshLoader.getLocalNodeIDsFromGlobal(self.mesh['ids'],nastranOrdering=True))
        if len(set(self.to_tacs))!=len(self.nodes) or not np.allclose(self.nodes[self.to_tacs],self.mesh['nodes']):
            raise RuntimeError('TACS/Nastran ordering audit failed')

    def modes(self,mode):
        for c in self.cons:
            c.setFailureModes(includePanelMaterialFailure=mode!='buckling',includeStiffenerMaterialFailure=mode!='buckling',
              includeLocalBuckling=mode!='material',includeGlobalBuckling=mode!='material',
              includeStiffenerColumnBuckling=mode!='material',includeStiffenerCrippling=mode!='material')

    def solve(self,data,pressure=False):
        t=time.perf_counter(); p=self.problem; self.modes('all')
        p.setDesignVars(np.asarray(data.get('design',self.base_dv),dtype=TACS.dtype))
        nodes=np.asarray(data.get('nodes',self.mesh['nodes']),dtype=TACS.dtype)
        ordered=np.empty_like(nodes); ordered[self.to_tacs]=nodes; p.setNodes(ordered.ravel())
        p.zeroLoads(); p.addInertialLoad(np.array(data.get('inertia_vector',[0.,0.,-G*float(data.get('load_factor',1.))]),dtype=TACS.dtype))
        external=np.zeros((len(nodes),6),dtype=TACS.dtype)
        if pressure:
            lower=[r['comp'] for r in self.panels if r['name'].startswith('L_SKIN')]
            # Official BDF lower-skin orientation: negative TACS pressure is upward/inward.
            p.addPressureToComponents(lower,-30000.)
        else:
            external[self.to_tacs]=data['loads']
        p.solve(Fext=external.ravel())
        funcs={}; p.evalFunctions(funcs)
        u=p.getVariables().reshape(-1,6)[self.to_tacs]
        residual=np.zeros(u.size,dtype=TACS.dtype); p.getResidual(residual,Fext=external.ravel())
        out={k.removeprefix('stw_'):float(v) for k,v in funcs.items()}
        out.update(wall_seconds=time.perf_counter()-t,residual_norm=float(np.linalg.norm(residual)),
                   max_translation_m=float(np.linalg.norm(u[:,:3],axis=1).max()))
        if data.get('ultimate',False):
            p.setVariables(p.getVariables()*1.5); f={}; p.evalFunctions(f,evalFuncs=['failure'])
            out['ultimate_failure']=float(f['stw_failure']); p.setVariables((u[np.argsort(self.to_tacs)]).ravel())
        if data.get('separate_modes',False):
            for mode in ('material','buckling'):
                self.modes(mode); f={}; p.evalFunctions(f,evalFuncs=['failure']); out[mode+'_KS']=float(f['stw_failure'])
            self.modes('all')
        if data.get('sensitivities',False):
            sens={}; p.evalFunctionsSens(sens,evalFuncs=['mass','failure'])
            out['mass_grad']=np.asarray(sens['stw_mass']['struct']).tolist()
            out['failure_grad']=np.asarray(sens['stw_failure']['struct']).tolist()
        if not np.isfinite(u).all() or not np.isfinite(list(funcs.values())).all() or out['mass']<=0:
            raise RuntimeError('Nonfinite/invalid TACS result')
        return out,u

    def optimize(self,data):
        """All panels sized independently under frozen maneuver loads, genuine adjoints.

        This is the inner step; outer aeroelastic/mission checks must be repeated.
        """
        nodes=data['nodes']; dv=data['design'].copy(); idx=np.array([r['dv_start']+k for r in self.panels for k in range(3)])
        scales=np.tile([100.,20.,100.],len(self.panels)); x0=dv[idx]*scales
        bounds=[(.0006*100,.05*100),(.0254*20,.149*20),(.0006*100,.025*100)]*len(self.panels)
        adjacency=[]
        for family in ('U_SKIN','L_SKIN','SPARS/SPAR.00','SPARS/SPAR.01'):
            rows=[i for i,r in enumerate(self.panels) if r['name'].startswith(family)]
            rows.sort(key=lambda i:self.panels[i]['centroid'][1]); adjacency += list(zip(rows[:-1],rows[1:]))
        def geometric(x):
            a=(x/scales).reshape(-1,3); t,h,ts=a.T
            cs=[15*t-ts,30*ts-h,h-5*ts,.15-h]
            for i,j in adjacency:
                delta=a[i]-a[j]; lim=np.array([.0025,.01,.0025]); cs.extend([lim-delta,lim+delta])
            return np.concatenate([np.atleast_1d(c) for c in cs])*100
        # These constraints are linear, so one exact finite difference builds their constant Jacobian.
        zero=np.zeros_like(x0); g0=geometric(zero)
        gj=np.column_stack([geometric(np.eye(len(x0))[j])-g0 for j in range(len(x0))])
        memo={}; hist=[]; feasible=[]
        def values(x):
            key=x.tobytes()
            if key in memo:return memo[key]
            design=dv.copy(); design[idx]=x/scales; rows=[]
            for j,n in enumerate((2.5,-1.)):
                r,_=self.solve(dict(nodes=nodes,design=design,loads=1.5*data[f'loads_{j}'],
                                  load_factor=n*1.5,inertia_vector=1.5*data.get(f'inertia_{j}',np.array([0.,0.,-n*G])),sensitivities=True))
                rows.append(r)
            mass=rows[0]['mass']; fail=np.array([r['failure'] for r in rows])
            v=(mass/1000.,np.array(rows[0]['mass_grad'])[idx]/scales/1000.,
               .95-fail,-np.stack([r['failure_grad'] for r in rows])[:,idx]/scales[None])
            memo[key]=v; hist.append({'mass':mass,'failure':fail.tolist()})
            if max(fail)<=.95001 and geometric(x).min()>=-1e-6:feasible.append((mass,x.copy()))
            if len(hist)%10==0:print('INNER',len(hist),'mass',mass,'failure',fail,flush=True)
            return v
        # Establish a feasible seed rather than accepting a failed optimizer iterate.
        if np.min(values(x0)[2])<0:
            for factor in (1.15,1.35,1.6,1.9):
                trial=x0.copy(); sizing=(trial/scales).reshape(-1,3)
                sizing[:,[0,2]]*=factor
                sizing[:,1]=np.maximum(sizing[:,1],5.1*sizing[:,2])
                trial=sizing.ravel()*scales
                if max(trial-np.array(bounds)[:,1])>0:continue
                if np.min(values(trial)[2])>=0:
                    x0=trial;break
        if not feasible:raise RuntimeError('No feasible structural seed found within frozen bounds')
        x=x0.copy(); move=.1; accepted=0; stop='iteration_budget'; iterations=0
        for iterations in range(int(data.get('max_iterations',24))):
            val,grad,cons,jac=values(x); g=geometric(x)
            span=move*np.maximum(abs(x),.2)
            b=np.array(bounds); db=list(zip(np.maximum(b[:,0]-x,-span),np.minimum(b[:,1]-x,span)))
            lp=linprog(grad,A_ub=np.vstack([-jac,-gj]),b_ub=np.r_[cons,g],bounds=db,method='highs',options={'threads':1})
            if not lp.success:
                move*=.5
                if move<1e-4:stop='linear_subproblem_failed';break
                continue
            improved=False
            for line in range(10):
                trial=x+lp.x*(.5**line); tv=values(trial)
                if min(tv[2])>=-1e-5 and geometric(trial).min()>=-1e-6 and tv[0]<val-1e-8:
                    x=trial;accepted+=1;improved=True;break
            if improved:move=min(.15,move*1.1)
            else:
                move*=.5
                if move<2e-4:stop='no_accepted_feasible_descent';break
        values(x)
        chosen=min(feasible,key=lambda item:item[0])[1]; dv[idx]=chosen/scales
        summary={'success':True,'message':stop,'iterations':int(iterations+1),
                 'algorithm':'bounded sequential linear programming with exact TACS gradients and feasible backtracking',
                 'accepted_steps':accepted,'optimality_certified':False,
                 'mass_kg':values(chosen)[0]*1000,'ultimate_failure':(.95-values(chosen)[2]).tolist(),
                 'minimum_geometric_margin_scaled':float(geometric(chosen).min()),'history':hist,
                 'selected_independently_feasible':True,'feasible_iterates':len(feasible)}
        return summary,dv

    def gradient_check(self,data):
        dv=data['design'].copy(); records=[]
        base,_=self.solve(dict(data,sensitivities=True))
        for slot in (0,1,2,3,4,5,6,7,8):
            direction=np.zeros(len(dv))
            if slot<3:
                for panel in self.panels:direction[panel['dv_start']+slot]=dv[panel['dv_start']+slot]
            else:
                panel=self.panels[(slot*17)%len(self.panels)]; j=panel['dv_start']+(slot%3);direction[j]=dv[j]
            h=1e-4
            plus,_=self.solve(dict(data,design=dv+h*direction,sensitivities=False))
            minus,_=self.solve(dict(data,design=dv-h*direction,sensitivities=False))
            for name in ('mass','failure'):
                fd=(plus[name]-minus[name])/(2*h); adj=float(np.dot(base[name+'_grad'],direction))
                records.append(dict(slot=slot,function=name,finite_difference=fd,adjoint=adj,
                         relative_error=abs(fd-adj)/max(abs(fd),1e-8)))
        return {'checks':records,'passed':max(r['relative_error'] for r in records)<.01},dv


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bdf',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); args.output.mkdir(exist_ok=True,parents=True)
    s=Structure(args.bdf)
    np.savez(args.output/'mesh.npz',nodes=s.mesh['nodes'],design=s.base_dv,conn=s.mesh['conn'],prop=s.mesh['prop'])
    (args.output/'panels.json').write_text(json.dumps(s.panels,indent=2))
    r,u=s.solve({'load_factor':2.5,'separate_modes':True},pressure=True)
    tip=s.mesh['nodes'][:,1]>13.99
    xyz=s.mesh['nodes']; top=tip & (xyz[:,2]>0)
    ii=np.where(top)[0]; ends=ii[[np.argmin(xyz[ii,0]),np.argmax(xyz[ii,0])]]
    r['tip_corners']=ends.tolist(); r['tip_deflection_m']=float(u[ends,2].mean())
    r['tip_twist_deg']=float(np.rad2deg(np.arctan2(u[ends[0],2]-u[ends[1],2],xyz[ends[1],0]-xyz[ends[0],0])))
    (args.output/'structural_benchmark.json').write_text(json.dumps(r,indent=2)); np.save(args.output/'structural_benchmark_u.npy',u)
    print('@@READY',flush=True)
    for line in sys.stdin:
        try:
            q=json.loads(line)
            if q.get('command')=='exit': break
            data=dict(np.load(q['input'],allow_pickle=False)); data.update(q.get('options',{}))
            dst=Path(q['output'])
            if q.get('command')=='gradient_check':
                r,dv=s.gradient_check(data); np.savez(dst,design=dv)
            elif q.get('command')=='optimize':
                r,dv=s.optimize(data); np.savez(dst,design=dv)
            else:
                r,u=s.solve(data); np.savez(dst,displacements=u)
            dst.with_suffix('.json').write_text(json.dumps(r,indent=2,allow_nan=False))
            print('@@DONE '+str(dst),flush=True)
        except Exception as exc:
            traceback.print_exc(); print('@@ERROR '+str(exc),flush=True)

if __name__=='__main__': main()
