"""Portable interactive 3-D CFD/FM inspection, using only saved real arrays."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np


def write_validation_report(summary, output):
    output = Path(output)
    base = summary["results"]["pretrained"]
    adapted = summary["results"]["adapted"]
    lookup = {c["sample_id"]: c for c in adapted["heldout_cases"]}
    cases = []
    for case in base["heldout_cases"]:
        other = lookup[case["sample_id"]]
        with np.load(output / "pretrained" / case["array_file"], allow_pickle=False) as a, \
             np.load(output / "adapted" / other["array_file"], allow_pickle=False) as b:
            # Preserve wing surface connectivity. Downsampling affects display only.
            span = np.unique(np.r_[np.arange(0, a["truth"].shape[1], 4), a["truth"].shape[1]-1])
            chord = np.unique(np.r_[np.arange(0, a["truth"].shape[2], 4), a["truth"].shape[2]-1])
            surface = lambda v: np.round(v[:, span][:, :, chord], 5).tolist()
            geometry = a["geometry"]
            span_positions = geometry[2].mean(axis=1)
            span_fractions = (span_positions - span_positions.min()) / max(float(np.ptp(span_positions)), 1e-12)
            slices = [int(np.argmin(abs(span_fractions - eta))) for eta in (.25, .5, .75)]
            sections = []
            for row in slices:
                x = geometry[0, row]
                local = (x - x.min()) / max(float(np.ptp(x)), 1e-12)
                sections.append({"eta": round(float(span_fractions[row]), 3), "x": local.tolist(),
                                 "truth": a["truth"][0, row].tolist(),
                                 "pretrained": a["prediction"][0, row].tolist(),
                                 "adapted": b["prediction"][0, row].tolist()})
            cases.append({"sample_id": case["sample_id"], "shape_id": case["shape_id"],
                          "condition": case["condition"], "geometry": surface(geometry),
                          "truth": surface(a["truth"])[0], "pretrained": surface(a["prediction"])[0],
                          "adapted": surface(b["prediction"])[0], "sections": sections,
                          "reference": case["reference_coefficients"],
                          "base_coefficients": case["coefficients"], "adapted_coefficients": other["coefficients"],
                          "base_gate": case["gate"], "adapted_gate": other["gate"]})
    payload = {"cases": cases, "summary": {"base": base["metrics"], "adapted": adapted["metrics"],
                "base_assessment": base["heldout_assessment"], "adapted_assessment": adapted["heldout_assessment"],
                "base_calibration": base["calibration"], "adapted_calibration": adapted["calibration"],
                "protocol": summary["protocol"], "training": summary["training"]}}
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    page = _TEMPLATE.replace("__DATA__", data)
    (output / "report.html").write_text(page, encoding="utf-8")
    return output / "report.html"


_TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CFD × Foundation Model · 独立验证</title>
<style>
*{box-sizing:border-box}body{margin:0;color:#dce5ee;background:#0b1220;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:36px 30px}h1{font-size:32px;letter-spacing:-1px;margin:8px 0}h2{font-size:20px;margin:0 0 16px}p{margin:8px 0;color:#a6b5c8}.eyebrow{color:#65d0d0;letter-spacing:2px;font-size:12px}.pill{padding:5px 12px;background:#213247;border-radius:18px;font-size:12px;display:inline-block}.note{background:#30291c;border-left:3px solid #e7b864;padding:12px 18px;color:#ecdbb9;margin:22px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}.card,.panel{background:#111e30;border:1px solid #263549;border-radius:12px;padding:20px}.value{font-size:28px;font-weight:650;color:#f1f6fc}.label{font-size:12px;color:#95abc3}.delta{font-size:12px;color:#70ceca}.toolbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:16px}select,button{background:#203449;color:#e9f3ff;border:1px solid #4a657e;border-radius:7px;padding:8px 12px}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.view{position:relative;border:1px solid #314356;border-radius:10px;overflow:hidden;background:#0b1523}.view header{padding:12px 16px;color:#d5e5f7;font-weight:600}.view canvas{width:100%;height:300px;display:block;touch-action:none;cursor:grab}.legend{display:flex;align-items:center;gap:12px;font-size:12px;color:#abc0d5;margin:16px 0}.ramp{width:200px;height:10px;background:linear-gradient(90deg,#285bd0,#41c5e8,#f4f3d7,#ec9353,#bb244b)}.row{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:18px 0}.chart{width:100%;height:280px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:right;padding:10px;border-bottom:1px solid #2c3e52}td:first-child,th:first-child{text-align:left}.gate{padding:12px;background:#243549;border-radius:8px;margin:12px 0;word-break:break-word}.gate b{color:#e7b864}details{margin-top:20px}pre{white-space:pre-wrap;max-height:350px;overflow:auto;font-size:12px;background:#09111c;padding:14px}.good{color:#75d5ac}.bad{color:#edb176}footer{margin-top:24px;color:#7993b0;font-size:12px}a{color:#76cfd5}@media(max-width:900px){.cards{grid-template-columns:1fr 1fr}.views,.row{grid-template-columns:1fr}main{padding:20px}}
</style><main>
<div class="eyebrow">AERODYNAMIC SURROGATE · INDEPENDENT VALIDATION</div>
<h1>CFD 与 Foundation Model：先看精度，再谈加速</h1>
<p>真实预训练权重 → 少量机翼微调 → 独立校准 → 本次适配未见几何测试。拖动机翼可同步旋转三个视图。</p>
<div class="note">当前对照为公开 CRMpert CFD 标签；没有新跑 CFD，也没有实测 CFD 加速倍数。置信区间只校准 CL/CD；未校准压力场、载荷或梯度。拒绝使用 FM 时显示“需要 CFD”，不会把标签读取伪装成求解器回退。</div>
<div class="cards" id="cards"></div>
<section class="panel"><div class="toolbar"><h2 style="margin:0">01 · 三维机翼压力对照</h2><select id="case"></select><span id="condition" class="pill"></span><button id="reset">重置视角</button></div>
<div class="views"><div class="view"><header>CFD 参考 · published labels</header><canvas id="truth"></canvas></div><div class="view"><header>原始 FM · pretrained</header><canvas id="pretrained"></canvas></div><div class="view"><header id="adaptation-label">适配 FM</header><canvas id="adapted"></canvas></div></div>
<div class="legend"><span>Cp = −1.5</span><span class="ramp"></span><span>+0.8</span><span>三个视图共享色标；超出色标的值仅在显示时截断。</span></div>
<div id="coefs"></div><div id="gate" class="gate"></div>
</section>
<div class="row"><section class="panel"><h2>02 · 翼剖面 Cp / x/c</h2><div class="toolbar"><select id="section"><option value="0">展向约 25%</option><option value="1" selected>展向约 50%</option><option value="2">展向约 75%</option></select><span class="label">蓝：CFD　橙：原始 FM　青：适配 FM</span></div><canvas class="chart" id="sectionplot"></canvas></section>
<section class="panel"><h2>03 · 全部测试几何的阻力误差</h2><p>每组两根柱：原始 FM / 适配 FM。虚线为事先设定的误差容差。</p><canvas class="chart" id="errors"></canvas></section></div>
<section class="panel"><h2>04 · 置信度与证据范围</h2><div id="coverage"></div><p>目标覆盖率是在交换性假设下对几何组的边际覆盖；不保证任意一个机翼，也不保证筛选后接受子集的覆盖率。输入范围检查只是粗略防线，不是完整的分布外检测器。划分隔离针对本次适配；未逐几何审计上游预训练集重叠。</p><details><summary>查看固定实验协议、校准与训练记录</summary><pre id="metadata"></pre></details></section>
<footer>数据：<a href="https://huggingface.co/datasets/thuerey-group/CRMpert">CRMpert, CC-BY-SA-4.0</a> · 模型：<a href="https://github.com/tum-pbs/AeroTransformer">AeroTransformer</a> · 所有评分使用完整数组；三维视图仅作降采样显示。原始记录见同目录 summary.json、split.json 和模型子目录。</footer>
</main><script>
const DATA=__DATA__;
const $=id=>document.getElementById(id);let selected=0,az=-0.82,el=0.69;const names=['truth','pretrained','adapted'];
function fmt(n,d=4){return Number.isFinite(n)?n.toFixed(d):'未测量'}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function canvas(id){let e=$(id),r=e.getBoundingClientRect(),d=window.devicePixelRatio||1;e.width=r.width*d;e.height=r.height*d;let c=e.getContext('2d');c.scale(d,d);return [c,r.width,r.height]}
const colors=[[40,91,208],[65,197,232],[244,243,215],[236,147,83],[187,36,75]];
function color(v){let t=Math.max(0,Math.min(.9999,(v+1.5)/2.3))*4,i=Math.floor(t),f=t-i;return 'rgb('+colors[i].map((x,k)=>Math.round(x+(colors[i+1][k]-x)*f)).join(',')+')'}
function drawWing(id){let [c,w,h]=canvas(id),a=DATA.cases[selected],g=a.geometry,rows=g[0].length,cols=g[0][0].length;
 let pts=[],mins=[Infinity,Infinity,Infinity],maxs=[-Infinity,-Infinity,-Infinity];for(let i=0;i<rows;i++)for(let j=0;j<cols;j++)for(let k=0;k<3;k++){mins[k]=Math.min(mins[k],g[k][i][j]);maxs[k]=Math.max(maxs[k],g[k][i][j])}
 let mid=mins.map((v,k)=>(v+maxs[k])/2),extent=Math.max(...maxs.map((v,k)=>v-mins[k]));
 function proj(i,j){let x=(g[0][i][j]-mid[0])/extent,y=(g[2][i][j]-mid[2])/extent,z=(g[1][i][j]-mid[1])/extent;let X=x*Math.cos(az)-y*Math.sin(az),Y=x*Math.sin(az)+y*Math.cos(az);return [X,Y*Math.cos(el)-z*Math.sin(el),Y*Math.sin(el)+z*Math.cos(el)]}
 let maxX=0,maxY=0;for(let i=0;i<rows;i++){pts[i]=[];for(let j=0;j<cols;j++){let p=proj(i,j);pts[i][j]=p;maxX=Math.max(maxX,Math.abs(p[0]));maxY=Math.max(maxY,Math.abs(p[1]))}}
 let scale=Math.min((w-40)/(2*maxX||1),(h-40)/(2*maxY||1)),quads=[];for(let i=0;i<rows-1;i++)for(let j=0;j<cols-1;j++){let p=[pts[i][j],pts[i+1][j],pts[i+1][j+1],pts[i][j+1]];quads.push({p,z:p.reduce((s,q)=>s+q[2],0)/4,v:(a[id][i][j]+a[id][i+1][j]+a[id][i+1][j+1]+a[id][i][j+1])/4})}
 quads.sort((a,b)=>a.z-b.z);for(let q of quads){c.beginPath();q.p.forEach((p,k)=>k?c.lineTo(w/2+p[0]*scale,h/2+p[1]*scale):c.moveTo(w/2+p[0]*scale,h/2+p[1]*scale));c.closePath();c.fillStyle=color(q.v);c.fill();c.strokeStyle=c.fillStyle;c.lineWidth=.4;c.stroke()}}
function plotSection(){let [c,w,h]=canvas('sectionplot'),s=DATA.cases[selected].sections[+$('section').value],v=[...s.truth,...s.pretrained,...s.adapted],lo=Math.min(-1.2,...v)-.1,hi=Math.max(.8,...v)+.1;
 let X=x=>45+x*(w-65),Y=y=>20+(y-lo)/(hi-lo)*(h-55);c.strokeStyle='#40556c';c.beginPath();c.moveTo(45,20);c.lineTo(45,h-35);c.lineTo(w-20,h-35);c.stroke();c.fillStyle='#9eb1c6';c.font='11px system-ui';for(let k=0;k<=4;k++){let y=lo+(hi-lo)*k/4;c.fillText(y.toFixed(1),5,Y(y)+4);c.fillText((k/4).toFixed(2),X(k/4)-10,h-15)}
 ['truth','pretrained','adapted'].forEach((n,k)=>{c.strokeStyle=['#79aaff','#f1a866','#69d5cc'][k];c.lineWidth=k===0?2:1.5;c.beginPath();s.x.forEach((x,i)=>i?c.lineTo(X(x),Y(s[n][i])):c.moveTo(X(x),Y(s[n][i])));c.stroke()});c.fillText('Cp ↓',5,12);c.fillText('x/c',w-25,h-3)}
function plotErrors(){let [c,w,h]=canvas('errors'),items=DATA.cases,tol=DATA.summary.protocol.config.tolerances.CD*10000,values=items.map(a=>[Math.abs(a.base_coefficients.CD-a.reference.CD)*10000,Math.abs(a.adapted_coefficients.CD-a.reference.CD)*10000]),mx=Math.max(tol,...values.flat())*1.15;let Y=v=>h-35-v/mx*(h-60),dx=(w-60)/items.length;
 c.strokeStyle='#40556c';c.beginPath();c.moveTo(40,15);c.lineTo(40,h-35);c.lineTo(w-15,h-35);c.stroke();values.forEach((v,i)=>v.forEach((x,k)=>{c.fillStyle=k?'#69d5cc':'#f1a866';c.fillRect(43+i*dx+k*dx*.4,Y(x),dx*.36,h-35-Y(x))}));c.strokeStyle='#e6cb88';c.setLineDash([4,4]);c.beginPath();c.moveTo(40,Y(tol));c.lineTo(w-15,Y(tol));c.stroke();c.setLineDash([]);c.font='11px system-ui';c.fillStyle='#bac9d7';c.fillText(tol+' counts',w-85,Y(tol)-5);c.fillText('0',20,h-30);c.fillText(mx.toFixed(1),5,20);c.fillText('测试几何（原始顺序）',w/2-65,h-10)}
function update(){let a=DATA.cases[selected];$('condition').textContent=`shape ${a.shape_id} · Mach ${fmt(a.condition.mach,3)} · α ${fmt(a.condition.alpha_deg,2)}°`;
 $('coefs').innerHTML='<table><tr><th>气动系数</th><th>CFD 参考</th><th>原始 FM</th><th>适配 FM</th><th>适配绝对误差</th></tr>'+['CL','CD','CM'].map(n=>`<tr><td>${n}${n==='CM'?'（诊断）':''}</td><td>${fmt(a.reference[n],5)}</td><td>${fmt(a.base_coefficients[n],5)}</td><td>${fmt(a.adapted_coefficients[n],5)}</td><td>${fmt(Math.abs(a.adapted_coefficients[n]-a.reference[n]),5)}</td></tr>`).join('')+'</table>';
 let gate=a.adapted_gate,reasonNames={calibrated_interval_exceeds_tolerance:'校准误差界超过预设容差',outside_training_support:'输入超出训练数据的经验范围'};
 $('gate').innerHTML='<b>当前机翼：'+(gate.accepted?'允许使用 FM 的 CL/CD':'需要 CFD 复核')+'</b><p>'+esc((gate.reasons||[]).map(x=>reasonNames[x]||x).join('；'))+'</p><span class="label">本阶段只记录路由决定，尚未执行求解器回退。</span>';names.forEach(drawWing);plotSection();plotErrors()}
DATA.cases.forEach((a,i)=>{$('case').add(new Option('测试机翼 '+a.shape_id+' / 样本 '+a.sample_id,i))});
$('adaptation-label').textContent='适配 FM · '+(DATA.summary.protocol.config.adaptation.mode==='head'?'冻结主干，训练输出层':'全参数微调');
let b=DATA.summary.base,a=DATA.summary.adapted,change=(b.CD_drag_counts_mae-a.CD_drag_counts_mae)/b.CD_drag_counts_mae*100;
$('cards').innerHTML=[['独立测试',a.shapes+' 个机翼','训练与校准几何均不重叠'],['适配后 CD 平均误差',fmt(a.CD_drag_counts_mae,2)+' counts','原始 '+fmt(b.CD_drag_counts_mae,2)+'；变化 '+fmt(change,1)+'%（正值为降低）'],['适配后 Cp RMSE',fmt(a.Cp_mean_rmse,4),'原始 '+fmt(b.Cp_mean_rmse,4)],['推理中位耗时',fmt(a.median_prediction_seconds*1000,1)+' ms','包含准备、模型与系数积分；非 CFD 加速倍数']].map(x=>`<div class="card"><div class="label">${x[0]}</div><div class="value">${x[1]}</div><div class="delta">${x[2]}</div></div>`).join('');
let ba=DATA.summary.base_assessment,aa=DATA.summary.adapted_assessment,bc=DATA.summary.base_calibration,ac=DATA.summary.adapted_calibration;
$('coverage').innerHTML='<table><tr><th>检查项</th><th>原始 FM</th><th>适配 FM</th></tr>'+[
 ['目标边际覆盖率',fmt(bc.nominal_marginal_group_coverage*100,0)+'%',fmt(ac.nominal_marginal_group_coverage*100,0)+'%'],
 ['测试集实际覆盖率',fmt(ba.observed_simultaneous_geometry_group_coverage*100,1)+'%',fmt(aa.observed_simultaneous_geometry_group_coverage*100,1)+'%'],
 ['校准 CD 区间半宽',fmt(bc.interval_half_widths.CD*10000,2)+' counts',fmt(ac.interval_half_widths.CD*10000,2)+' counts'],
 ['预设 CD 误差容差',fmt(bc.tolerances.CD*10000,1)+' counts',fmt(ac.tolerances.CD*10000,1)+' counts'],
 ['允许使用 FM 的测试样本',ba.accepted_case_count+' / '+ba.case_count,aa.accepted_case_count+' / '+aa.case_count],
 ['需要 CFD 的测试样本',ba.fallback_requested_case_count,aa.fallback_requested_case_count]
 ].map(r=>'<tr>'+r.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</table>';
$('metadata').textContent=JSON.stringify(DATA.summary,null,2);$('case').onchange=()=>{selected=+$('case').value;update()};$('section').onchange=plotSection;$('reset').onclick=()=>{az=-.82;el=.69;update()};
names.forEach(n=>{let e=$(n),last=null;e.onpointerdown=v=>{last=[v.clientX,v.clientY];e.setPointerCapture(v.pointerId)};e.onpointermove=v=>{if(!last)return;az+=(v.clientX-last[0])*.01;el+=(v.clientY-last[1])*.01;last=[v.clientX,v.clientY];names.forEach(drawWing)};e.onpointerup=e.onpointercancel=()=>last=null});window.addEventListener('resize',update);update();
</script></html>'''
