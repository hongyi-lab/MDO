"""STW hybrid FM-cruise/VLM-maneuver aeroelastic runtime with real TACS.

This explicitly does not pretend that the frozen transonic FM supports the
low-Mach maneuver conditions. No low-Mach inputs are replaced or clipped.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .stw_task import atmosphere, Transfer, section_change, G, mission, wingbox_volume, read_stw_bdf, sizing_margins
from .input_contract import encode_stw_surface, encode_condition
from .aerotransformer import AeroTransformerPredictor, _add_upstream_paths


class Worker:
    def __init__(self,project,code,output,bdf):
        self.project,self.code,self.output=map(Path,(project,code,output)); self.output.mkdir(parents=True,exist_ok=True)
        self.count=0; self.log=(self.output/'worker.log').open('w')
        def inside(p):return '/home/mdolabuser/mount/'+str(Path(p).relative_to(self.project))
        self.inside=inside
        cmd=[sys.executable,str(self.project/'cfd/run_cfd_rootless.py'),'--project',str(self.project),'--',
             '/bin/bash','-c','source "$BASHRC_MDOLAB"; exec "$@"','stw','python',inside(self.code/'scripts/stw_tacs_worker.py'),
             '--bdf',inside(bdf),'--output',inside(self.output)]
        self.p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        self.wait('@@READY'); f=np.load(self.output/'mesh.npz'); self.nodes=f['nodes']; self.design=f['design']
        self.panels=json.loads((self.output/'panels.json').read_text())
    def wait(self,token):
        while True:
            line=self.p.stdout.readline(); self.log.write(line); self.log.flush()
            if line.startswith(token): return
            if line.startswith('@@ERROR'): raise RuntimeError(line)
            if not line and self.p.poll() is not None: raise RuntimeError('TACS worker ended; see worker.log')
    def request(self,data,options=None,command='solve'):
        self.count+=1; a=self.output/'request.npz'; b=self.output/'response.npz'; np.savez(a,**data)
        self.p.stdin.write(json.dumps(dict(input=self.inside(a),output=self.inside(b),options=options or {},command=command))+'\n'); self.p.stdin.flush()
        self.wait('@@DONE'); f=np.load(b); arrays={k:f[k].copy() for k in f.files}; f.close()
        return json.loads(b.with_suffix('.json').read_text()),arrays
    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write('{"command":"exit"}\n'); self.p.stdin.flush(); self.p.wait(timeout=30)
        self.log.close()


class FM:
    def __init__(self,checkpoint):
        self.model=AeroTransformerPredictor(checkpoint,'cuda'); self.calls=0
    def evaluate(self,vertices,alpha,mach=.77):
        self.calls+=1
        sample=encode_stw_surface(vertices,root_chord_m=5.,reference_half_area_m2=45.5)
        sample['condition']=encode_condition(alpha_deg=alpha,mach=mach)
        pred=self.model.predict(sample); c=pred['coefficients']
        if abs(pred['cp']).max()>100 or not np.isfinite(list(c.values())).all() or c['CD']<=0:
            raise ValueError('FM numerical guard failed; no field clipping permitted')
        from cfdpost.wing.basic import BasicWing
        from cfdpost.utils import get_dxyforce_2d
        wing=BasicWing(paras={'ref_area':45.5/25},aoa=float(alpha),iscentric=True,normal_factors=(1.,150.,300.))
        wing.read_formatted_surface(geometry=sample['original_geometry'].copy(),data=pred['fields'].copy(),isinitg=False,isnormed=True)
        geom,field=wing.surface_blocks[0]; cp=field[...,wing.store_variables['cp']]
        forces=get_dxyforce_2d(geom,cp,wing.cf_vector)[...,[0,2,1]]*25
        angle=np.deg2rad(alpha); total=forces.sum(axis=(0,1)); cl=(-total[0]*np.sin(angle)+total[2]*np.cos(angle))/45.5
        cd=(total[0]*np.cos(angle)+total[2]*np.sin(angle))/45.5
        if not np.allclose([cl,cd],[c['CL'],c['CD']],rtol=1e-6,atol=1e-7):raise RuntimeError('FM spatial force integration mismatch')
        return dict(CL=float(cl),CD=float(cd),forces_per_q=forces,fields=pred['fields'])


def camber_mesh(native,nspan=25,nchord=7):
    """Actual RAE2822 camber, rather than an invented flat plate."""
    root=native[0]; leidx=root[:,0].argmin(); x0=root[leidx,0]; chord=np.ptp(root[:,0])
    xs=.5*(1-np.cos(np.linspace(0,np.pi,nchord))); ys=np.linspace(0,14,nspan)
    lo=root[:leidx+1][::-1]; up=root[leidx:255]
    cz=.5*(np.interp(xs,(lo[:,0]-x0)/chord,lo[:,2]/chord)+np.interp(xs,(up[:,0]-x0)/chord,up[:,2]/chord))
    le=native[np.arange(len(native)),native[:,:,0].argmin(axis=1)]
    p=np.zeros((nchord,nspan,3)); p[:,:,0]=np.interp(ys,le[:,1],le[:,0])[None]+xs[:,None]*(5.-.25*ys)
    p[:,:,1]=ys; p[:,:,2]=cz[:,None]*(5.-.25*ys)
    return p


class LowSpeedVLM:
    def __init__(self,mesh):
        import openmdao.api as om
        from openaerostruct.aerodynamics.geometry import VLMGeometry
        from openaerostruct.aerodynamics.compressible_states import CompressibleVLMStates
        self.calls=0; m=mesh[:,::-1].copy(); m[:,:,1]*=-1
        surf=dict(name='wing',mesh=m,symmetry=True,S_ref_type='projected')
        p=om.Problem(reports=False)
        p.model.add_subsystem('geom',VLMGeometry(surface=surf),promotes_inputs=[('def_mesh','mesh')])
        p.model.add_subsystem('states',CompressibleVLMStates(surfaces=[surf]),promotes_inputs=['alpha','beta','rho','v','Mach_number',('wing_def_mesh','mesh')])
        p.model.connect('geom.normals','states.wing_normals'); p.model.set_input_defaults('mesh',m,units='m')
        p.setup(); self.problem=p
    def evaluate(self,mesh,alpha,mach=.4577):
        self.calls+=1; m=mesh[:,::-1].copy(); m[:,:,1]*=-1
        p=self.problem; atm=atmosphere(0); v=mach*atm['a']; q=.5*atm['rho']*v*v
        for k,val in dict(mesh=m,rho=atm['rho'],v=v,Mach_number=mach).items():p.set_val(k,val)
        # PGTransform promotes radians, unlike the usual AeroPoint degree interface.
        p.set_val('alpha',alpha,units='deg'); p.set_val('beta',0.,units='deg')
        p.run_model()
        forces=p.get_val('states.wing_sec_forces').copy(); points=p.get_val('states.collocation_points.force_pts').copy().reshape(forces.shape)
        forces[:,:,1]*=-1; points[:,:,1]*=-1
        forces=forces[:,::-1]; points=points[:,::-1]
        total=forces.sum(axis=(0,1)); a=np.deg2rad(alpha)
        cl=(-total[0]*np.sin(a)+total[2]*np.cos(a))/(q*45.5)
        cd=(total[0]*np.cos(a)+total[2]*np.sin(a))/(q*45.5)
        if not np.isfinite(forces).all():raise RuntimeError('Nonfinite VLM force')
        return dict(CL=float(cl),CD=float(cd),forces=forces,points=points)


class Coupled:
    def __init__(self,project,code,output,native,bdf):
        self.native=native; self.output=Path(output)
        self.worker=Worker(project,code,self.output/'structure',bdf)
        self.mesh=read_stw_bdf(bdf); self.fm=FM(Path(project)/'models/AeroTransformer/ATsurf_L')
        self.camber=camber_mesh(native); self.vlm=LowSpeedVLM(self.camber)
        self.rows=[]; self.design_count=0

    def geometry(self,twist,thickness,design,camber_delta=(0.,0.,0.)):
        n=section_change(self.worker.nodes,self.native,twist,thickness,camber_delta)
        s=section_change(self.native,self.native,twist,thickness,camber_delta)
        c=section_change(self.camber,self.native,twist,thickness,camber_delta)
        d=design.copy()
        for panel in self.worker.panels:
            ids=np.unique(self.mesh['conn'][self.mesh['prop']==panel['prop']]); x=n[ids]
            if 'SKIN' not in panel['name']:d[panel['dv_start']+3]=np.ptp(x[:,2])
        return n,s,c,d

    def analyze(self,twist,thickness,design,*,camber_delta=(0.,0.,0.),tol=3e-4,maxiter=70,fixed_alpha=None):
        started=time.perf_counter(); nodes,surface,camber,d=self.geometry(twist,thickness,design,camber_delta)
        # Obtain structural mass (geometry and self-weight included) before weight/trim closure.
        mass,_=self.worker.request(dict(nodes=nodes,design=d,loads=np.zeros((len(nodes),6))),{'load_factor':0.})
        mb=mass['mass']; volume=wingbox_volume(self.mesh,nodes)
        reports={}; saved={}
        for name,nfactor in [('cruise',1.),('pullup',2.5),('pushdown',-1.)]:
            cruise=name=='cruise'; jig=surface if cruise else camber
            vmap=Transfer(nodes,jig.reshape(-1,3))
            if cruise:
                centers=.25*(jig[:-1,:-1]+jig[:-1,1:]+jig[1:,:-1]+jig[1:,1:])
            else:
                bound=.75*jig[:-1]+.25*jig[1:]; centers=.5*(bound[:,:-1]+bound[:,1:])
            fmap=Transfer(nodes,centers.reshape(-1,3)); u=np.zeros((len(nodes),6))
            a=(4.8 if cruise else (9. if nfactor>0 else -6.)); previous=None; history=[]
            atm=atmosphere(10400 if cruise else 0); mach=.77 if cruise else .4577
            q=.5*atm['rho']*(mach*atm['a'])**2
            converged=False
            for k in range(maxiter):
                deformed=jig+vmap.motion(u).reshape(jig.shape)
                if cruise:
                    aero=self.fm.evaluate(deformed,float(a)); force=aero['forces_per_q']*q
                    m=mission(mb,aero['CL'],aero['CD']); target=m['mid_cruise_kg']*G/(2*q*45.5)
                else:
                    aero=self.vlm.evaluate(deformed,float(a)); force=aero['forces']
                    lgm=41500+2*10.147*mb**.8162; target=nfactor*lgm*G/(2*q*45.5)
                loads=fmap.loads(force.reshape(-1,3))
                angle=np.deg2rad(a); inertia=nfactor*G*np.array([np.sin(angle),0.,-np.cos(angle)])
                st,arr=self.worker.request(dict(nodes=nodes,design=d,loads=loads,inertia_vector=inertia),{'load_factor':nfactor,'ultimate':True})
                unew=arr['displacements']; du=float(np.max(np.abs(unew[:,:3]-u[:,:3])))
                err=float(aero['CL']-target); row=dict(iteration=k,alpha=float(a),CL=aero['CL'],target_CL=float(target),
                  CD=aero['CD'],displacement_residual_m=du,CL_residual=err,tip_max=float(unew[:,2].max()),failure=st['failure'])
                history.append(row)
                if k%10==0:print('COUPLING',name,k,'alpha',a,'dU',du,'dCL',err,flush=True)
                if du<tol and ((fixed_alpha is not None and cruise) or abs(err)<tol*.15):
                    converged=True; u=unew; break
                # Coupled trim and displacement fixed point. No pressure scaling to force CL equality.
                anew=float(np.clip(a-.65*err/.105,2.,12.) if cruise else np.clip(a-.65*err/.095,-18.,18.))
                if fixed_alpha is not None and cruise:anew=float(fixed_alpha)
                u=.55*u+.45*unew; a=anew
            reports[name]=dict(converged=converged,alpha_deg=float(a),CL=aero['CL'],CD=aero['CD'],
                target_CL=float(target),CL_residual=err,displacement_residual_m=du,iterations=len(history),
                mass_box_kg=mb,KS_limit=st['failure'],ultimate_failure=st['ultimate_failure'],
                max_deflection_m=st['max_translation_m'],history=history,
                aerodynamic_backend='ATsurf_L' if cruise else 'OpenAeroStruct_2.12.0_compressible_VLM')
            saved[name]=dict(nodes=nodes,design=d,loads=loads,displacements=u,jig=jig,deformed=deformed,
                             forces=force,points=centers,alpha=a,inertia_vector=inertia,fields=aero.get('fields',np.zeros(0)))
            print('CONDITION',name,'converged',converged,'iterations',k+1,'ultimate',st['ultimate_failure'],flush=True)
            if not converged:
                # Preserve precise failure stage. Never present nonconverged aeroelastic points as scores.
                break
        m=mission(mb,reports['cruise']['CL'],reports['cruise']['CD'])
        capacity=804.*(2*.85*volume+2.763)
        result=dict(mission=m,wingbox_volume_m3=volume,fuel_capacity_kg=capacity,
                    fuel_capacity_margin_kg=capacity-m['total_fuel_kg'],conditions=reports,
                    coupled_converged=len(reports)==3 and all(r['converged'] for r in reports.values()),
                    twist=list(map(float,twist)),thickness=list(map(float,thickness)),wall_seconds=time.perf_counter()-started)
        result['camber_delta']=list(map(float,camber_delta))
        result['sizing_margins_m']=sizing_margins(d,self.worker.panels)
        result['geometry_bounds_satisfied']=bool(max(abs(np.asarray(twist)))<=4.000000001 and min(thickness)>=.95-1e-9 and max(thickness)<=1.5+1e-9 and max(abs(np.asarray(camber_delta)))<=.020000001)
        result['feasible_model_only']=bool(result['coupled_converged'] and result['fuel_capacity_margin_kg']>=-1e-3
                    and result['geometry_bounds_satisfied'] and min(result['sizing_margins_m'].values())>=-1e-8
                    and max(reports[k]['ultimate_failure'] for k in ('pullup','pushdown'))<=1.0001)
        return result,saved

    def close(self):self.worker.close()
