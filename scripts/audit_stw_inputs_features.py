#!/usr/bin/env python3
"""No-training server audit: official loader/model parity and frozen features."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mdo_demo.io import read_json, write_json, sha256_file
from mdo_demo.input_contract import encode_condition, encode_stw_surface, decode_stw_vertices, describe_contract


def loader_parity(project, output):
    from mdo_demo.dataset import CRMpertDataset
    from flowvae.dataset import MCFlowDataset
    data=CRMpertDataset(project/'data/assets/CRMpertTrainDev')
    samples=[data.sample(i) for i in (0,1)]
    fixture=output/'official_loader_fixture'; fixture.mkdir()
    np.save(fixture/'data.npy',np.stack([s['fields'] for s in samples]))
    np.save(fixture/'geom.npy',np.stack([s['geometry'] for s in samples]))
    np.save(fixture/'index.npy',np.array(data.index[:2],copy=True))
    np.savetxt(fixture/'indices.txt',[0,1],fmt='%d')
    official=MCFlowDataset(['data','geom','index'],is_ref=False,d_c=2,
        split_paras={'indexFileName':str(fixture/'indices.txt')},
        aux_channel_take=[2,3],data_base=str(fixture),marker_idx=2)
    rows=[]
    for i,s in enumerate(samples):
        item=official[i]
        g=item['input'].numpy(); c=item['aux'].numpy()
        rows.append({'sample_id':s['sample_id'],'condition':c.tolist(),
            'geometry_equal':bool(np.array_equal(g,s['geometry'])),
            'condition_equal':bool(np.array_equal(c,s['condition'])),
            'geometry_max_abs_difference':float(np.max(abs(g-s['geometry'])))})
    if not all(r['geometry_equal'] and r['condition_equal'] for r in rows):
        raise ValueError('Official dataset loader parity failed')
    return rows,samples[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True); p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True)
    args=p.parse_args(); project=args.project.resolve(); output=args.output.resolve()
    import torch
    from mdo_demo.aerotransformer import AeroTransformerPredictor, _add_upstream_paths
    _add_upstream_paths()
    from flowvae.app.wing.models import AeroTransformer
    import flowvae.dataset as official_dataset
    import flowvae.base_model.pdet.pde_transformer as pde_module
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True)
    protocol=read_json(args.protocol)
    if sha256_file(args.input) != protocol['input_sha256']: raise ValueError('Input hash mismatch')
    with (project/'manifests/compute.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        load=os.getloadavg(); mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        ram=int(mem['MemAvailable'].split()[0])/1024**2
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
        disk=os.statvfs(project).f_bavail*os.statvfs(project).f_frsize/1024**3
        if load[0]>24 or ram<8 or disk<3 or int(gpu.split(',')[-1].split()[0])>30: raise RuntimeError('Shared resources busy')
        output.mkdir(parents=True,exist_ok=False)
        report={'status':'running','protocol_sha256':sha256_file(args.protocol),'input_contract':describe_contract(),
                'new_training':False,'cfd_executed':False,'resources':{'cpu_load':load,'available_RAM_GiB':ram,'free_disk_GiB':disk,'gpu':gpu,'threads':1},
                'official_source_sha256':{m.__name__:sha256_file(m.__file__) for m in [official_dataset,pde_module]},'models':{}}
        with np.load(args.input,allow_pickle=False) as f:
            # Reference CL/CD deliberately not read in this feature extraction job.
            native=np.array(f['native_vertices_m']); alpha=np.array(f['alpha'])
            geometry=encode_stw_surface(native,root_chord_m=5.,reference_half_area_m2=45.5)
            equal={k:bool(np.array_equal(geometry[k],f[k])) for k in geometry}
        if not all(equal.values()): raise ValueError('Explicit contract changed v1 input unexpectedly')
        report['contract_v1_tensor_equal']=equal
        report['native_roundtrip_max_abs_m']=float(abs(decode_stw_vertices(geometry['original_geometry'],root_chord_m=5.)-native).max())
        report['official_dataset_parity'],control=loader_parity(project,output)
        ckpts={'pretrained':project/'models/AeroTransformer/ATsurf_L',
               'adapted':Path((project/'manifests/model_pilot_run.path').read_text().strip())/'adapted_checkpoint'}
        for name,ckpt in ckpts.items():
            started=time.perf_counter(); adapter=AeroTransformerPredictor(ckpt,'cuda')
            config=read_json(ckpt/'model_config')
            direct=AeroTransformer(**config['init_kwargs']).cuda().eval()
            direct.load_state_dict(torch.load(ckpt/'best_model_weights',map_location='cuda',weights_only=True),strict=True)
            parity=[]
            for label,sample in [('released_CRM_control',control)]+[(f'STW_alpha_{a}',dict(geometry,condition=encode_condition(alpha_deg=a,mach=.77))) for a in [0.,2.,5.]]:
                wrapped=adapter.predict(sample)['fields']
                with torch.inference_mode():
                    raw=direct(torch.as_tensor(np.array(sample['geometry'],copy=True),device='cuda').unsqueeze(0),
                               code=torch.as_tensor(np.array(sample['condition'],copy=True),device='cuda').unsqueeze(0))[0][0].cpu().numpy()
                parity.append({'case':label,'bitwise_equal':bool(np.array_equal(raw,wrapped)),
                               'max_abs_difference':float(abs(raw-wrapped).max()),'raw_Cp_absmax':float(abs(raw[0]).max())})
            if not all(r['bitwise_equal'] for r in parity): raise ValueError('Direct official model parity failed')
            del direct; gc.collect(); torch.cuda.empty_cache()
            trace=[]; features=[]; pooled={}; events=[]; handles=[]
            last_cond=f'c_embedder_{adapter.model.num_encoder_layers}'
            def hook(key):
                def capture(module,inp,out):
                    a=out.detach()
                    events.append({'module':key,'shape':list(a.shape),'absmax':float(a.abs().max()),'finite':bool(torch.isfinite(a).all())})
                    if key=='latent': pooled[key]=a.mean(dim=(2,3))[0].cpu().numpy()
                    if key==last_cond: pooled[key]=a[0].cpu().numpy()
                return capture
            for key,module in adapter.model.named_children():
                if key in ['x_embedder','latent','final_layer'] or key.startswith(('c_embedder_','encoder_level_','decoder_level_')):
                    handles.append(module.register_forward_hook(hook(key)))
            fields=[]; coefficients=[]
            for a in alpha:
                events.clear(); pooled.clear()
                sample=dict(geometry,condition=encode_condition(alpha_deg=a,mach=.77))
                prediction=adapter.predict(sample)
                features.append(np.concatenate([pooled['latent'],pooled[last_cond]]))
                fields.append(prediction['fields']); coefficients.append([prediction['coefficients'][k] for k in ['CL','CD']])
                trace.append({'alpha_deg':float(a),'events':list(events)})
            for h in handles: h.remove()
            x=np.asarray(features)
            if not np.isfinite(x).all(): raise ValueError('Nonfinite extracted features')
            np.savez_compressed(output/f'{name}_features.npz',features=x,alpha=alpha,raw_CL_CD=coefficients,fields=fields)
            report['models'][name]={'direct_official_parity':parity,'feature_definition':protocol['feature_definition'],
                'feature_shape':list(x.shape),'trace':trace,'weights_sha256':sha256_file(ckpt/'best_model_weights'),
                'architecture_unchanged':True,'all_backbone_weights_frozen':True,'audit_wall_seconds':time.perf_counter()-started}
            write_json(output/'audit.json',report)
            print(name,'official parity PASS; features',x.shape,'Cp absmax',float(abs(np.array(fields)[:,0]).max()),flush=True)
            del adapter; gc.collect(); torch.cuda.empty_cache()
        report['status']='completed'; write_json(output/'audit.json',report)


if __name__=='__main__': main()
