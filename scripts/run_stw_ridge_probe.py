#!/usr/bin/env python3
"""Prespecified retrospective single-wing polar calibration, not MDO scoring."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mdo_demo.io import read_json,write_json,sha256_file
from mdo_demo.ridge_probe import select_ridge,predict_ridge,DEFAULT_ALPHAS
from mdo_demo.stw_benchmark import polar_scores


def freeze(input_file, protocol):
    if protocol.exists(): raise FileExistsError(protocol)
    protocol.parent.mkdir(parents=True,exist_ok=True)
    write_json(protocol,{'version':'stw_input_ridge_v2','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'input_sha256':sha256_file(input_file),'fit_indices':list(range(0,21,2)),
        'eval_indices':list(range(1,21,2)),'ridge_alphas':DEFAULT_ALPHAS.tolist(),
        'feature_definition':'concatenate spatial mean of final encoder latent output and final condition embedding; no learned feature selection',
        'baselines':['pretrained raw','adapted raw','condition-only cubic ridge'],
        'probes':['pretrained frozen feature ridge readout','adapted frozen feature ridge readout'],
        'hyperparameter_selection':'LOOCV on the 11 fit points only; refit standardization within each fold',
        'target_definition':'direct [CL,CD] readout; coefficients only, no corrected Cp/shear fields',
        'physical_cases_unchanged':True,'angle_clamping_or_offset':False,
        'prior_label_exposure':'All 21 public reference rows previously inspected; retrospective development validation, not blind test',
        'scope':'Interpolation on one fixed STW geometry and Mach; not shape generalization, constraints, or MDO design quality',
        'no_new_CFD':True,'backbone_training':False,'architecture_changes':False})


def fit_and_report(input_file, protocol_path, features_dir, output):
    protocol=read_json(protocol_path); audit=read_json(features_dir/'audit.json')
    if audit['status']!='completed' or audit['protocol_sha256']!=sha256_file(protocol_path): raise ValueError('Unqualified feature audit')
    if sha256_file(input_file)!=protocol['input_sha256']: raise ValueError('Input identity mismatch')
    output.mkdir(parents=True,exist_ok=False)
    with np.load(input_file,allow_pickle=False) as f:
        a=f['alpha']; y=np.column_stack([f['reference_CL'],f['reference_CD']])
    tr=np.array(protocol['fit_indices']); ev=np.array(protocol['eval_indices'])
    if set(tr)&set(ev) or sorted(np.r_[tr,ev].tolist())!=list(range(len(a))): raise ValueError('Invalid frozen split')
    methods={}; rows=[]; fits={}
    features={'condition_only_cubic':np.column_stack([a,a*a,a*a*a])}
    for name in ['pretrained','adapted']:
        with np.load(features_dir/f'{name}_features.npz',allow_pickle=False) as f:
            if not np.array_equal(a,f['alpha']): raise ValueError('Feature row mismatch')
            features[name+'_frozen_feature_ridge']=np.array(f['features'])
            methods[name+'_raw']={'predictions':np.array(f['raw_CL_CD'])}
    for name,x in features.items():
        t0=time.perf_counter(); fit,cv=select_ridge(x[tr],y[tr],np.array(protocol['ridge_alphas']))
        fit_time=time.perf_counter()-t0
        prediction=predict_ridge(fit,x)
        # Readout timing alone, explicitly excluding backbone inference.
        for _ in range(20): predict_ridge(fit,x[ev[:1]])
        timing=[]
        for _ in range(100):
            started=time.perf_counter(); predict_ridge(fit,x[ev[:1]]); timing.append(time.perf_counter()-started)
        np.savez_compressed(output/f'{name}_fit.npz',**fit)
        fits[name]=cv
        methods[name]={'predictions':prediction,'fit_LOOCV_wall_seconds':fit_time,
            'readout_only_seconds_median':float(np.median(timing)),'CV':cv}
    for name,item in methods.items():
        prediction=item.pop('predictions')
        item['eval_10']=polar_scores(prediction[ev],y[ev]); item['fit_11']=polar_scores(prediction[tr],y[tr])
        item['eval_negative_CD_count']=int((prediction[ev,1]<0).sum())
        for i in range(len(a)):
            rows.append({'method':name,'index':i,'role':'fit' if i in tr else 'eval','alpha_deg':a[i],
                'CL_reference':y[i,0],'CD_reference':y[i,1],'CL_prediction':prediction[i,0],'CD_prediction':prediction[i,1],
                'CL_difference':prediction[i,0]-y[i,0],'CD_difference_counts':1e4*(prediction[i,1]-y[i,1])})
    with (output/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'protocol':protocol,'protocol_sha256':sha256_file(protocol_path),'feature_audit_sha256':sha256_file(features_dir/'audit.json'),
        'n_geometries':1,'n_fit':len(tr),'n_eval':len(ev),'methods':methods,
        'interpretation':'Retrospective held-out polar interpolation against public CFD reference, not physical ground truth or blind MDO score',
        'unresolved':'surface sampling and omitted physical conditioning unchanged; ridge CL/CD cannot validate structural loads'}
    write_json(output/'summary.json',summary)
    with (output/'scores.csv').open('w',newline='') as f:
        records=[dict(method=k,**v['eval_10'],negative_CD=v['eval_negative_CD_count']) for k,v in methods.items()]
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    for name,v in methods.items(): print(name,v['eval_10'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','fit'])
    p.add_argument('--input',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--features',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.mode=='freeze': freeze(args.input,args.protocol)
    else: fit_and_report(args.input,args.protocol,args.features,args.output)
