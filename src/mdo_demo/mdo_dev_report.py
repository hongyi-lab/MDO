"""Portable, exploratory DEV comparison of saved real model evaluations.

Only display arrays are downsampled/quantized. Metrics are copied from the
full-resolution evaluation records. No test, calibration, gate, or CFD speedup
claim is synthesized by this reporter.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np


def _pack(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Display arrays must be nonempty and finite")
    offset = float(array.min())
    scale = float(np.ptp(array)) / 65535.0
    quantized = (np.rint((array - offset) / scale).astype("<u2") if scale > 0
                 else np.zeros(array.shape, dtype="<u2"))
    return {"shape": list(array.shape), "offset": offset, "scale": scale,
            "values": base64.b64encode(quantized.tobytes()).decode("ascii")}


def _artifact(evaluation_path, relative):
    root = Path(evaluation_path).resolve().parent
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Array file must stay inside its evaluation directory")
    return path


def _provenance(report, path):
    return {"evaluation_file": Path(path).name,
            "evaluation_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "protocol_id": report.get("protocol_id"),
            "partition": report["partition"],
            "split_sha256": report["split_sha256"],
            "dataset_manifest_sha256": report["dataset_manifest_sha256"],
            "device": report.get("device"), "host": report.get("host"),
            "timestamp_utc": report.get("timestamp_utc"),
            "timing_scope": report.get("timing_scope"),
            "warmup": report.get("warmup"),
            "model_loading_s": report.get("model_loading_s"),
            "prediction_timing": report.get("prediction_timing"),
            "component_timing": report.get("component_timing"),
            "model": report.get("provenance", {})}


def _load_curves(before, after):
    keys = ("spanwise_lift_prediction", "spanwise_lift_truth", "spanwise_width")
    if not all(key in before and key in after for key in keys):
        return None
    truth, truth_other = before[keys[1]], after[keys[1]]
    width, width_other = before[keys[2]], after[keys[2]]
    if not np.array_equal(truth, truth_other) or not np.array_equal(width, width_other):
        raise ValueError("Before/after spanwise truth and widths differ")
    if (truth.ndim != 1 or width.shape != truth.shape or not np.isfinite(width).all()
            or np.any(width <= 0) or not np.isclose(width.sum(), 1.0, rtol=1e-5, atol=1e-6)):
        raise ValueError("Invalid saved span-load density/normalized strip widths")
    if before[keys[0]].shape != truth.shape or after[keys[0]].shape != truth.shape:
        raise ValueError("Saved span-load prediction shapes differ")
    eta = np.cumsum(width) - width / 2
    return _pack(np.stack((eta, truth, before[keys[0]], after[keys[0]])))


def write_mdo_dev_report(before_evaluation_path, after_evaluation_path, output_html):
    """Write one offline HTML file; return its Path.

    Both JSON files must contain partition='dev', matching split/dataset hashes,
    the same unique sample IDs, macro_metrics, and cases pointing to real NPZs
    with geometry/truth/prediction. Optional NPZ spanwise arrays are documented
    in _load_curves. All cases are retained, including deteriorated predictions.
    """
    before_path, after_path = Path(before_evaluation_path), Path(after_evaluation_path)
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    for report in (before, after):
        if report.get("partition") != "dev":
            raise ValueError("Only explicitly marked DEV evaluations are supported")
        if report.get("mode") != "real_checkpoint_dataset_evaluation":
            raise ValueError("A DEV report requires real checkpoint/dataset evaluations")
        if not isinstance(report.get("macro_metrics"), dict) or not report.get("cases"):
            raise ValueError("DEV evaluation must contain macro_metrics and nonempty cases")
    for key in ("split_sha256", "dataset_manifest_sha256"):
        if not before.get(key) or before[key] != after.get(key):
            raise ValueError(f"Before/after {key} must match")
    before_ids = [str(case["sample_id"]) for case in before["cases"]]
    after_ids = [str(case["sample_id"]) for case in after["cases"]]
    if len(before_ids) != len(set(before_ids)) or len(after_ids) != len(set(after_ids)):
        raise ValueError("Duplicate source sample IDs")
    if set(before_ids) != set(after_ids):
        raise ValueError("Before/after source sample IDs differ")
    other_cases = {str(case["sample_id"]): case for case in after["cases"]}
    cases = []
    for original in before["cases"]:
        adapted = other_cases[str(original["sample_id"])]
        if (str(original["shape_id"]) != str(adapted["shape_id"])
                or original["condition"] != adapted["condition"]
                or original["reference_coefficients"] != adapted["reference_coefficients"]):
            raise ValueError("Before/after geometry, condition, or coefficient reference differs")
        with np.load(_artifact(before_path, original["array_file"]), allow_pickle=False) as a, \
                np.load(_artifact(after_path, adapted["array_file"]), allow_pickle=False) as b:
            geometry, truth = a["geometry"], a["truth"]
            if (truth.ndim != 3 or truth.shape[0] != 3 or geometry.shape != truth.shape
                    or a["prediction"].shape != truth.shape or b["prediction"].shape != truth.shape):
                raise ValueError("Expected matching cell-center geometry and three-channel fields")
            if not np.array_equal(geometry, b["geometry"]) or not np.array_equal(truth, b["truth"]):
                raise ValueError("Before/after CFD geometry or truth arrays differ")
            span = np.unique(np.linspace(0, truth.shape[1] - 1, min(32, truth.shape[1]), dtype=int))
            chord = np.unique(np.linspace(0, truth.shape[2] - 1, min(64, truth.shape[2]), dtype=int))
            surface = lambda value: value[span][:, chord]
            section_z = geometry[2].mean(axis=1)
            eta = (section_z - section_z.min()) / max(float(np.ptp(section_z)), 1e-12)
            sections = []
            for requested in (0.25, 0.5, 0.75):
                row = int(np.argmin(np.abs(eta - requested)))
                x = geometry[0, row]
                local_x = (x - x.min()) / max(float(np.ptp(x)), 1e-12)
                sections.append({"eta": float(eta[row]), "data": _pack(np.stack((
                    local_x, truth[0, row], a["prediction"][0, row], b["prediction"][0, row])))})
            cases.append({
                "sample_id": str(original["sample_id"]), "shape_id": str(original["shape_id"]),
                "condition": original["condition"],
                "geometry": [_pack(surface(channel)) for channel in geometry],
                "truth": _pack(surface(truth[0])),
                "pretrained": _pack(surface(a["prediction"][0])),
                "adapted": _pack(surface(b["prediction"][0])),
                "sections": sections, "loads": _load_curves(a, b),
                "reference": original["reference_coefficients"],
                "reintegrated_reference": original.get("reference_reintegrated"),
                "pretrained_coefficients": original["coefficients"],
                "adapted_coefficients": adapted["coefficients"],
                "pretrained_field_errors": original.get("field_errors"),
                "adapted_field_errors": adapted.get("field_errors"),
                "pretrained_prediction_s": original.get("end_to_end_prediction_s"),
                "adapted_prediction_s": adapted.get("end_to_end_prediction_s"),
            })
    cases.sort(key=lambda case: ((0, int(case["sample_id"])) if case["sample_id"].isdigit()
                                 else (1, case["sample_id"])))
    initial = max(range(len(cases)), key=lambda i: abs(cases[i]["adapted_coefficients"]["CD"] - cases[i]["reference"]["CD"]))
    payload = {"cases": cases, "initial_index": initial,
               "initial_selection_rule": "Largest adapted absolute CD error across all DEV samples; ties use ascending source sample ID",
               "display_encoding": "uint16 linear quantization; surface at most 32x64; full chordwise Cp sections; metrics remain full precision",
               "metrics": {"pretrained": before["macro_metrics"], "adapted": after["macro_metrics"]},
               "provenance": {"pretrained": _provenance(before, before_path), "adapted": _provenance(after, after_path),
                              "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    page = _TEMPLATE.replace("__DATA__", encoded)
    if len(page.encode("utf-8")) > 20 * 1024 * 1024:
        raise ValueError("Portable report exceeds 20 MiB; refuse silently omitting DEV cases")
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page, encoding="utf-8")
    return output_html


_TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MDO adaptation · DEV exploratory</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b1422;color:#e7edf4;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1560px;margin:auto;padding:28px}.eyebrow{color:#f4b774;font-size:12px;letter-spacing:2px}h1{font-size:32px;line-height:1.2;margin:12px 0}h2{font-size:19px;margin:0 0 12px}p{color:#a9bdd0;margin:8px 0}.warning{margin:20px 0;border-left:4px solid #f0b367;background:#302a21;padding:15px 18px;color:#f2dbb6}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.card,.panel{background:#122135;border:1px solid #2a4057;border-radius:12px;padding:20px}.panel{margin:16px 0}.label,.caption{font-size:12px;color:#a3b9ce}.value{font-size:26px;font-weight:650}.good{color:#70d0bb}.bad{color:#ffb17c}.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.toolbar h2{margin:0}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}.view{border:1px solid #324b65;border-radius:9px;overflow:hidden;background:#0c1828}.view header{padding:10px 14px;font-weight:600;font-size:14px}.view canvas{display:block;width:100%;height:320px;cursor:grab;touch-action:none}.ramp{width:200px;height:10px;background:linear-gradient(90deg,#285bd0,#41c5e8,#f4f3d7,#ec9353,#bb244b);display:inline-block;margin:0 12px}.legend{color:#aec0d2;font-size:12px;margin:12px 0}.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}.row .panel{margin-top:0}.chart{display:block;width:100%;height:285px}.bars{height:260px;cursor:crosshair}select,button{background:#20374e;color:#edf5ff;border:1px solid #4b6783;padding:8px 12px;border-radius:6px;max-width:100%}button{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:10px;text-align:right;border-bottom:1px solid #29405a}th:first-child,td:first-child{text-align:left}.tablewrap{overflow:auto}details{margin-top:16px}summary{cursor:pointer;color:#a9d9dd}pre{white-space:pre-wrap;overflow:auto;max-height:380px;background:#091320;padding:16px;font-size:12px}.footer{font-size:12px;color:#8da8bf;margin:24px 0}.empty{padding:70px 15px;text-align:center;color:#8fa7bd}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}@media(max-width:950px){main{padding:18px}.cards{grid-template-columns:repeat(2,1fr)}.views,.row{grid-template-columns:1fr}.view canvas{height:280px}h1{font-size:27px}}
</style><main>
<div class="eyebrow">DEV EXPLORATORY · CFD-FM-v2 · METHOD DEVELOPMENT ONLY</div>
<h1>让 FM 学会关注升阻力和载荷分布</h1>
<p>公开 CFD 参考场 / 原始预训练模型 / MDO 目标适配模型。拖动任一机翼，同步旋转三个视图。</p>
<div class="warning"><b>这是 Dev 探索性比较，不是独立测试结论。</b>这些样本用于方法开发；报告不提供校准覆盖率、可信门控或 CFD 加速倍数。当前 CFD 图来自已发布的数据标签，没有把读取标签当成一次新 CFD 求解。</div>
<div class="cards" id="cards"></div>
<section class="panel"><div class="toolbar"><h2>01 · 同一个机翼，同一色标</h2><select id="case" aria-label="Select source sample"></select><button id="reset">重置视角</button></div>
<p id="condition"></p><p class="caption" id="selection"></p>
<div class="views"><div class="view"><header>CFD · 已发布参考场</header><canvas id="truth"></canvas></div><div class="view"><header id="before-label">原始 FM · pretrained</header><canvas id="pretrained"></canvas></div><div class="view"><header id="after-label">MDO 目标适配 · adapted</header><canvas id="adapted"></canvas></div></div>
<div class="legend">Cp = −1.5 <span class="ramp"></span> +0.8 · 共享色标，范围之外仅截断颜色；几何与压力降采样只影响显示。</div><div class="tablewrap" id="coefficients"></div>
</section>
<div class="row"><section class="panel"><div class="toolbar"><h2>02 · 剖面压力 Cp</h2><select id="section"><option value="0">展向 25%</option><option value="1" selected>展向 50%</option><option value="2">展向 75%</option></select></div><p class="legend"><span style="color:#7dabff">蓝：CFD</span>　<span style="color:#f2ad72">橙：原始 FM</span>　<span style="color:#70d0bb">青：MDO 适配</span></p><canvas class="chart" id="sections"></canvas></section>
<section class="panel"><h2>03 · 展向升力载荷</h2><p class="caption">无量纲 dCL/dη；与训练同一表面积分，不代表 FEM 结构载荷传递。</p><canvas class="chart" id="loads"></canvas><div id="load-note" class="caption"></div></section></div>
<section class="panel"><div class="toolbar"><h2>04 · 全部 Dev 样本的 CD 误差</h2><select id="order"><option value="source">按源样本 ID</option><option value="error">按适配后误差，从大到小</option></select></div><p class="caption">每组橙 / 青柱为原始 / 适配绝对误差，单位 drag counts（1 count = 10⁻⁴）。全部样本均保留；点击柱形可切换机翼，黄色标记为当前样本。</p><canvas class="chart bars" id="errors"></canvas></section>
<section class="panel"><h2>05 · 全精度指标与证据范围</h2><p class="caption">先在每个几何内平均工况，再给每个几何相同权重。负的改善率表示退步；所有指标直接来自 evaluation.json，不用显示用的低精度数组重新评分。</p><div class="tablewrap" id="metrics"></div><details><summary>查看完整 macro_metrics</summary><pre id="metrics-json"></pre></details><details><summary>查看权重、源代码、数据划分与计时来源</summary><pre id="metadata"></pre></details></section>
<p class="footer">数据：CRMpert / AeroTransformer authors，CC-BY-SA-4.0。场图为全部 Dev 样本的离线交互视图；压力面最多 32×64，剖面保留原始点数；显示数组使用 16 位线性量化。完整数组、全精度评分及可复核记录仍在源 evaluation.json / case_*.npz 中。此页面不请求任何外部脚本、字体或网络接口。</p>
</main><script>
const DATA=__DATA__;
const $=id=>document.getElementById(id),names=['truth','pretrained','adapted'],colors=['#7dabff','#f2ad72','#70d0bb'];
let selected=DATA.initial_index,az=-.82,el=.69,decoded=null,barOrder=[];
function fmt(v,d=4){return Number.isFinite(v)?v.toFixed(d):'未保存'}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function unpack(p){let raw=atob(p.values),bytes=Uint8Array.from(raw,c=>c.charCodeAt(0)),view=new DataView(bytes.buffer),arr=[];for(let i=0;i<bytes.length/2;i++)arr.push(p.offset+p.scale*view.getUint16(2*i,true));if(p.shape.length===1)return arr;let w=p.shape[1];return Array.from({length:p.shape[0]},(_,i)=>arr.slice(i*w,(i+1)*w))}
function current(){if(!decoded||decoded.index!==selected){let a=DATA.cases[selected];decoded={index:selected,geometry:a.geometry.map(unpack),truth:unpack(a.truth),pretrained:unpack(a.pretrained),adapted:unpack(a.adapted),sections:a.sections.map(s=>({eta:s.eta,data:unpack(s.data)})),loads:a.loads?unpack(a.loads):null}}return decoded}
function canvas(id){let e=$(id),r=e.getBoundingClientRect(),ratio=window.devicePixelRatio||1;e.width=r.width*ratio;e.height=r.height*ratio;let c=e.getContext('2d');c.scale(ratio,ratio);return[c,r.width,r.height]}
const ramp=[[40,91,208],[65,197,232],[244,243,215],[236,147,83],[187,36,75]];
function color(v){let t=Math.max(0,Math.min(.99999,(v+1.5)/2.3))*4,i=Math.floor(t),f=t-i;return'rgb('+ramp[i].map((n,k)=>Math.round(n+(ramp[i+1][k]-n)*f)).join(',')+')'}
function drawWing(id){let[c,w,h]=canvas(id),a=current(),g=a.geometry,rows=g[0].length,cols=g[0][0].length,min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];for(let i=0;i<rows;i++)for(let j=0;j<cols;j++)for(let k=0;k<3;k++){min[k]=Math.min(min[k],g[k][i][j]);max[k]=Math.max(max[k],g[k][i][j])}let mid=min.map((v,k)=>(v+max[k])/2),extent=Math.max(1e-12,...max.map((v,k)=>v-min[k])),pts=[],mx=0,my=0;for(let i=0;i<rows;i++){pts[i]=[];for(let j=0;j<cols;j++){let x=(g[0][i][j]-mid[0])/extent,y=(g[2][i][j]-mid[2])/extent,z=(g[1][i][j]-mid[1])/extent,X=x*Math.cos(az)-y*Math.sin(az),Y=x*Math.sin(az)+y*Math.cos(az),p=[X,Y*Math.cos(el)-z*Math.sin(el),Y*Math.sin(el)+z*Math.cos(el)];pts[i][j]=p;mx=Math.max(mx,Math.abs(p[0]));my=Math.max(my,Math.abs(p[1]))}}let scale=Math.min((w-35)/(2*mx||1),(h-35)/(2*my||1)),quads=[];for(let i=0;i<rows-1;i++)for(let j=0;j<cols-1;j++){let p=[pts[i][j],pts[i+1][j],pts[i+1][j+1],pts[i][j+1]];quads.push({p,z:p.reduce((s,q)=>s+q[2],0)/4,v:(a[id][i][j]+a[id][i+1][j]+a[id][i+1][j+1]+a[id][i][j+1])/4})}quads.sort((a,b)=>a.z-b.z);for(let q of quads){c.beginPath();q.p.forEach((p,k)=>k?c.lineTo(w/2+p[0]*scale,h/2+p[1]*scale):c.moveTo(w/2+p[0]*scale,h/2+p[1]*scale));c.closePath();c.fillStyle=color(q.v);c.fill();c.strokeStyle=c.fillStyle;c.lineWidth=.5;c.stroke()}}
function lineChart(id,values,yname,negativeUp=false){let[c,w,h]=canvas(id);if(!values){c.fillStyle='#93aec7';c.font='14px system-ui';c.fillText('源记录未保存展向载荷数组',30,110);return}let x=values[0],series=values.slice(1),all=series.flat(),low=Math.min(...all),high=Math.max(...all),pad=Math.max((high-low)*.1,.02);low-=pad;high+=pad;let X=v=>52+v*(w-76),Y=v=>negativeUp?20+(v-low)/(high-low)*(h-57):h-37-(v-low)/(high-low)*(h-57);c.font='11px system-ui';for(let k=0;k<=4;k++){let v=low+(high-low)*k/4;c.strokeStyle='#243d56';c.beginPath();c.moveTo(52,Y(v));c.lineTo(w-24,Y(v));c.stroke();c.fillStyle='#afc2d5';c.fillText(v.toFixed(2),4,Y(v)+4);c.fillText((k/4).toFixed(2),X(k/4)-10,h-15)}c.strokeStyle='#637e99';c.beginPath();c.moveTo(52,20);c.lineTo(52,h-37);c.lineTo(w-24,h-37);c.stroke();series.forEach((s,j)=>{c.strokeStyle=colors[j];c.lineWidth=j===0?2.1:1.6;c.beginPath();s.forEach((v,i)=>i?c.lineTo(X(x[i]),Y(v)):c.moveTo(X(x[i]),Y(v)));c.stroke()});c.fillStyle='#bbcee0';c.fillText(yname,5,12);c.fillText(negativeUp?'x/c':'η',w-25,h-3)}
function drawSections(){let s=current().sections[+$('section').value];lineChart('sections',s.data,'Cp ↓',true)}
function error(a,key='adapted'){return Math.abs(a[key+'_coefficients'].CD-a.reference.CD)*10000}
function drawErrors(){let[c,w,h]=canvas('errors');barOrder=DATA.cases.map((_,i)=>i);if($('order').value==='error')barOrder.sort((i,j)=>error(DATA.cases[j])-error(DATA.cases[i]));let max=Math.max(1,...DATA.cases.map(a=>Math.max(error(a),error(a,'pretrained'))))*1.08,dx=(w-70)/barOrder.length,Y=v=>h-38-v/max*(h-65);c.font='11px system-ui';for(let k=0;k<=4;k++){let v=max*k/4;c.fillStyle='#a9bed3';c.fillText(v.toFixed(1),4,Y(v)+4);c.strokeStyle='#28415a';c.beginPath();c.moveTo(48,Y(v));c.lineTo(w-20,Y(v));c.stroke()}barOrder.forEach((index,i)=>{let a=DATA.cases[index];['pretrained','adapted'].forEach((key,k)=>{let value=error(a,key);c.fillStyle=colors[k+1];c.fillRect(49+i*dx+k*dx*.42,Y(value),Math.max(.35,dx*.38),h-38-Y(value))});if(index===selected){c.strokeStyle='#f5da8c';c.lineWidth=1;c.strokeRect(48+i*dx,16,Math.max(2,dx),h-53)}});let step=Math.max(1,Math.ceil(barOrder.length/8));for(let i=0;i<barOrder.length;i+=step){c.fillStyle='#adc2d7';c.fillText(DATA.cases[barOrder[i]].sample_id,49+i*dx-6,h-18)}c.fillText('source sample ID',w-130,h-2)}
function percent(before,after){return before>0?100*(before-after)/before:null}
function change(value){return value===null?'—':`<span class="${value>=0?'good':'bad'}">${value>=0?'+':''}${fmt(value,1)}%</span>`}
function metricTable(){let b=DATA.metrics.pretrained,a=DATA.metrics.adapted,rows=[['CL MAE',b.CL.mae,a.CL.mae,5],['CD MAE · counts',b.CD_drag_counts_mae,a.CD_drag_counts_mae,2],['CL 最大绝对误差',b.CL.max_abs,a.CL.max_abs,5],['CD 最大绝对误差 · counts',b.CD.max_abs*10000,a.CD.max_abs*10000,2],['Cp 平均 RMSE',b.Cp_mean_rmse,a.Cp_mean_rmse,5],['Cf_stream 平均 RMSE',b.Cf_stream_mean_rmse,a.Cf_stream_mean_rmse,6],['Cf_span 平均 RMSE',b.Cf_span_mean_rmse,a.Cf_span_mean_rmse,6]];for(let key of ['spanwise_lift_rmse','spanwise_lift_mean_rmse'])if(Number.isFinite(b[key])&&Number.isFinite(a[key]))rows.push(['展向载荷 RMSE',b[key],a[key],5]);$('metrics').innerHTML='<table><tr><th>Dev 指标</th><th>原始 FM</th><th>MDO 适配</th><th>改善率（正 = 降低）</th></tr>'+rows.map(r=>`<tr><td>${r[0]}</td><td>${fmt(r[1],r[3])}</td><td>${fmt(r[2],r[3])}</td><td>${change(percent(r[1],r[2]))}</td></tr>`).join('')+'</table>';$('cards').innerHTML=[['开发集',`${a.geometries} 个几何`,`${a.samples} 个工况 · 全部可选`],['适配后 CD MAE',`${fmt(a.CD_drag_counts_mae,2)} counts`,`原始 ${fmt(b.CD_drag_counts_mae,2)} · ${change(percent(b.CD_drag_counts_mae,a.CD_drag_counts_mae))}`],['适配后 Cp RMSE',fmt(a.Cp_mean_rmse,4),`原始 ${fmt(b.Cp_mean_rmse,4)}`],['推理中位耗时',`${fmt(a.median_prediction_seconds*1000,1)} ms`,`原始 ${fmt(b.median_prediction_seconds*1000,1)} ms；非 CFD 加速`]].map(r=>`<div class="card"><div class="label">${r[0]}</div><div class="value">${r[1]}</div><div class="caption">${r[2]}</div></div>`).join('')}
function update(){let a=DATA.cases[selected],d=current();$('case').value=String(selected);$('condition').textContent=`源样本 ${a.sample_id} · 几何 ${a.shape_id} · Mach ${fmt(a.condition.mach,3)} · α ${fmt(a.condition.alpha_deg,2)}°`;$('coefficients').innerHTML='<table><tr><th>系数</th><th>已发布参考</th><th>原始 FM</th><th>MDO 适配</th><th>适配绝对误差</th></tr>'+['CL','CD','CM'].map(k=>`<tr><td>${k}${k==='CM'?'（仅诊断）':''}</td><td>${fmt(a.reference[k],6)}</td><td>${fmt(a.pretrained_coefficients[k],6)}</td><td>${fmt(a.adapted_coefficients[k],6)}</td><td>${fmt(Math.abs(a.adapted_coefficients[k]-a.reference[k]),6)}</td></tr>`).join('')+'</table>';names.forEach(drawWing);drawSections();lineChart('loads',d.loads,'dCL/dη');$('load-note').textContent=d.loads?'曲线由完整网格积分后保存；CFD / 原始 FM / MDO 适配采用同一积分定义。':'未保存载荷曲线时不从降采样压力图伪造载荷。';drawErrors()}
DATA.cases.forEach((a,i)=>$('case').add(new Option(`源样本 ${a.sample_id} / 几何 ${a.shape_id}`,i)));
$('selection').textContent='初始显示规则：全部 Dev 样本中，适配后 CD 绝对误差最大的样本；并列按源样本 ID 排序。可在下拉框或柱形图查看全部样本。';
$('before-label').textContent='原始 FM · '+(DATA.provenance.pretrained.model.model||'pretrained');
$('after-label').textContent='MDO 目标适配 · '+(DATA.provenance.adapted.model.model||'adapted');
$('metrics-json').textContent=JSON.stringify(DATA.metrics,null,2);$('metadata').textContent=JSON.stringify({...DATA.provenance,display_encoding:DATA.display_encoding,initial_selection_rule:DATA.initial_selection_rule},null,2);
$('case').onchange=()=>{selected=+$('case').value;update()};$('section').onchange=drawSections;$('order').onchange=drawErrors;$('reset').onclick=()=>{az=-.82;el=.69;names.forEach(drawWing)};
$('errors').onclick=e=>{let r=e.currentTarget.getBoundingClientRect(),i=Math.floor((e.clientX-r.left-49)/(r.width-70)*barOrder.length);if(i>=0&&i<barOrder.length){selected=barOrder[i];update()}};
names.forEach(id=>{let canvas=$(id),last=null;canvas.onpointerdown=e=>{last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!last)return;az+=(e.clientX-last[0])*.01;el+=(e.clientY-last[1])*.01;last=[e.clientX,e.clientY];names.forEach(drawWing)};canvas.onpointerup=canvas.onpointercancel=()=>last=null});window.addEventListener('resize',update);metricTable();update();
</script></html>'''
