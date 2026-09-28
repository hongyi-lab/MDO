#!/usr/bin/env python3
"""Paper-style plots and OBJ of recorded FM shape optimization candidates."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mdo_demo.aero_contract import native_mainwing_vertices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run;out=args.output;out.mkdir(parents=True,exist_ok=False)
    result=json.loads((run/'result.json').read_text())
    if result['status'] not in ('completed','budget_limited'):raise ValueError('Run not complete')
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
         'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
         'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    selection=json.loads((run/'strict_selection.json').read_text())
    records=[result['initial'],selection['candidate']]
    labels=['Initial wing','FM-optimized candidate'];colors=['#5B6673','#CD703D']
    data=[]
    for name,record in zip(('initial','candidate_strict'),records):
        source=run/name
        native=native_mainwing_vertices(source/'aero/bundle/wing.xyz')
        with np.load(source/'aero/prediction.npz') as f:
            vertices=f['original_geometry'];cp=f['fields'][0]
        rot=np.asarray(record['frame_contract']['rotation_matrix_native_to_model'])
        physical=np.einsum('ab,bij->aij',np.diag([1,1,-1])@rot.T,vertices)
        data.append({'native':native,'physical':physical,'cp':cp,'record':record})

    def save(fig,name):
        for extension in ('png','svg','pdf'):fig.savefig(out/(name+'.'+extension),dpi=350)
        plt.close(fig)

    def panel(ax,d):
        g=d['physical'][:,::4,::4].transpose(1,2,0).copy()
        g=g[:,:,[2,0,1]];g[:,:,0]*=-1
        faces=np.stack([g[:-1,:-1],g[:-1,1:],g[1:,1:],g[1:,:-1]],axis=2).reshape(-1,4,3)
        vals=d['cp'].reshape(32,4,64,4).mean(axis=(1,3)).reshape(-1)
        ax.add_collection3d(Poly3DCollection(faces,facecolors=plt.cm.coolwarm(norm(vals)),edgecolors='none',alpha=1))
        ax.set(xlim=(.1,3.3),ylim=(-.05,2.8),zlim=(-.4,.6))
        ax.set_box_aspect((3.2,2.85,1),zoom=1.12);ax.view_init(elev=40,azim=135);ax.set_proj_type('ortho');ax.set_axis_off()

    norm=Normalize(-1.5,1.0)
    fig=plt.figure(figsize=(8.4,5.8))
    fig.text(.055,.954,'Wing shape optimization with FM + TACS',fontsize=14,weight='bold',color='#27323A')
    fig.text(.055,.916,'Adapted AeroTransformer Large | real shell FEA | independent CFD verification pending',fontsize=8,color='#596574')
    for i,d in enumerate(data):
        left=.035+.48*i
        fig.text(left+.035,.854,labels[i],fontsize=11,weight='bold',color=colors[i])
        ax=fig.add_axes([left,.40,.47,.46],projection='3d');panel(ax,d)
        r=d['record'];text=(f"Wingbox mass  {r['mass_kg']:.2f} kg\nTip displacement  {r['tip_displacement_m']*1000:.1f} mm\n"
                           f"KS yield index  {r['ks_failure']:.3f}\nLift coefficient  {r['coefficients']['CL']:.3f}")
        fig.text(left+.07,.390,text,fontsize=9,linespacing=1.55,color='#27323A')
        tag='FM-predicted constraints satisfied' if r['minimum_margin']>=0 else 'Violates at least one design constraint'
        fig.text(left+.07,.200,tag,fontsize=7.5,color='#52695E' if r['feasible_under_fm'] else '#AF513E')
    bar=fig.colorbar(ScalarMappable(norm=norm,cmap='coolwarm'),cax=fig.add_axes([.28,.125,.44,.018]),orientation='horizontal')
    bar.solids.set_rasterized(False);bar.outline.set_visible(False);bar.set_label('Predicted pressure coefficient, Cp (common scale)',fontsize=8)
    fig.text(.055,.038,f"10 variables | {result['physical_evaluation_count']} valid analyses | {result['optimization_wall_seconds']:.1f} s optimization wall time",fontsize=8,color='#596574')
    fig.text(.055,.012,'One-way, single-condition exploratory design. No buckling / flutter / aeroelastic feedback; not a certified final wing.',fontsize=7,color='#596574')
    save(fig,'fm_wing_before_after')

    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    good=[r for r in history if r['status']=='ok']
    fig,axes=plt.subplots(2,2,figsize=(8.4,6.0));fig.subplots_adjust(left=.085,right=.965,top=.87,bottom=.09,wspace=.32,hspace=.43)
    fig.suptitle('Geometry changes and optimization progress',x=.055,ha='left',fontsize=14,weight='bold',y=.975)
    fig.text(.055,.92,'Predictions from the adapted FM; feasibility refers to this model plus the common TACS solver.',fontsize=8,color='#596574')
    ax=axes[0,0]
    for d,label,color in zip(data,labels,colors):
        g=d['native'];span=-g[2].mean(axis=1)
        ax.plot(span,g[0].min(axis=1),color=color,lw=1.6,label=label)
        ax.plot(span,g[0].max(axis=1),color=color,lw=1.6)
        for j in (0,-1):ax.plot([span[j]]*2,[g[0].min(axis=1)[j],g[0].max(axis=1)[j]],color=color,lw=1.1)
    ax.set(xlabel='Native span coordinate (m)',ylabel='Streamwise coordinate (m)',title='a  Planform');ax.invert_yaxis();ax.legend(fontsize=7)
    ax=axes[0,1]
    for d,label,color in zip(data,labels,colors):
        g=d['native'].transpose(1,2,0);le=g[np.arange(len(g)),g[:,:,0].argmin(axis=1)];te=.5*(g[:,0]+g[:,-1])
        twist=-np.degrees(np.arctan2(te[:,1]-le[:,1],te[:,0]-le[:,0]))
        span=-g[:,:,2].mean(axis=1);eta=(span-span[0])/(span[-1]-span[0])
        ax.plot(eta,twist,color=color,lw=1.6,label=label)
    ax.set(xlabel='Fractional span',ylabel='Geometric nose-up twist (deg)',title='b  Twist distribution')
    ax=axes[1,0];running=np.inf;xs=[];ys=[]
    for r in good:
        if r['minimum_margin']>=0:running=min(running,r['mass_kg'])
        xs.append(r['cumulative_optimization_seconds']);ys.append(running if np.isfinite(running) else np.nan)
    ax.scatter([r['cumulative_optimization_seconds'] for r in good],[r['mass_kg'] for r in good],s=8,color='#B8C0C9',label='All evaluated designs')
    ax.step(xs,ys,where='post',color=colors[1],lw=1.6,label='Best strictly FM-feasible mass')
    ax.set(xlabel='Cumulative optimization time (s)',ylabel='Wingbox mass (kg)',title='c  Search history');ax.legend(fontsize=6.5)
    ax=axes[1,1];ax.plot([r['index'] for r in good],[r['minimum_margin'] for r in good],color=colors[1],lw=1.1)
    ax.axhline(0,color='#5B6673',lw=.8,ls='--')
    ax.set(xlabel='Design evaluation',ylabel='Minimum normalized constraint margin',title='d  Feasibility (nonnegative passes)')
    save(fig,'fm_shape_optimization_history')

    def obj(path,vertices,quads,note):
        with path.open('w') as f:
            f.write('# '+note+'\n# Native coordinates, metres. Surface/shell mesh, not a manufactured solid.\n')
            for p in vertices:f.write('v '+' '.join(f'{x:.12g}' for x in p)+'\n')
            for q in quads:f.write('f '+' '.join(str(int(x)+1) for x in q)+'\n')
    for name,d in zip(('initial','candidate'),data):
        g=d['native'].transpose(1,2,0);ns,nc=g.shape[:2]
        quads=np.array([[i*nc+j,i*nc+j+1,(i+1)*nc+j+1,(i+1)*nc+j] for i in range(ns-1) for j in range(nc-1)])
        obj(out/(name+'_wing_surface.obj'),g.reshape(-1,3),quads,'FM shape design; independent CFD verification pending')
        folder='candidate_strict' if name=='candidate' else name
        with np.load(run/folder/'structure/structure_arrays.npz') as a:
            obj(out/(name+'_wingbox.obj'),a['nodes_m'],a['quads'],'Actual undeformed TACS shell midsurface')
            obj(out/(name+'_deformed_wingbox_x1.obj'),a['nodes_m']+a['nodal_displacements_and_rotations'][:,:3],a['quads'],'Actual TACS displacement x1 under FM loads')
    rows=[]
    for label,r in zip(labels,records):
        rows.append({'design':label,'mass_kg':r['mass_kg'],'tip_mm':1000*r['tip_displacement_m'],
                     'ks_yield':r['ks_failure'],'CL_fixed_reference':r['coefficients']['CL'],'CD_fixed_reference':r['coefficients']['CD'],
                     'FM_predicted_feasible':r['minimum_margin']>=0,'CFD_verified':False,**r['design']})
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    shutil.copyfile(run/'history.csv',out/'history.csv');shutil.copyfile(run/'protocol.json',out/'protocol.json')
    shutil.copyfile(run/'result.json',out/'result.json');shutil.copyfile(run/'strict_selection.json',out/'strict_selection.json')
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    contract={'conclusion':'A bounded FM-driven shape/sizing search produced a recorded candidate; independent CFD validation is deferred.',
              'backend':'Python','archetype':'image plate + quantitative evidence','n':'one geometry family, one flow condition, one optimization start',
              'statistics':'No statistical error bars; points are correlated optimizer evaluations, not independent validation cases.',
              'Cp_display':'four-by-four cell means on downsampled vertices; common scale -1.5 to 1.0',
              'hardware':result['hardware'],'optimization_wall_seconds':result['optimization_wall_seconds'],
              'total_wall_seconds':result['total_wall_seconds'],'CFD_verified':False}
    (out/'figure_contract.json').write_text(json.dumps(contract,indent=2))
    print(json.dumps({'output':str(out),'mass_initial':records[0]['mass_kg'],'mass_candidate':records[1]['mass_kg'],
                      'candidate_FM_feasible':records[1]['feasible_under_fm']}))


if __name__=='__main__':main()
