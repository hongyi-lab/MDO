"""Self-contained local HTML report; no CDN, telemetry, or web service."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np


def number(value, digits=4):
    return "—" if value is None else f"{float(value):.{digits}g}"


def cp_plot(truth, predicted):
    """One mid-span surface strip. Index is explicit, not mislabeled x/c."""
    t = np.asarray(truth)[0]
    p = np.asarray(predicted)[0]
    if t.ndim != 2:
        raise ValueError("Cp chart expects a span-by-surface grid")
    row = t.shape[0] // 2
    t, p = t[row].reshape(-1), p[row].reshape(-1)
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(p))):
        raise ValueError("Non-finite chart values")
    lower, upper = min(t.min(), p.min()), max(t.max(), p.max())
    padding = max((upper - lower) * .08, .05)
    lower, upper = lower - padding, upper + padding
    width, height, left, top = 680, 260, 55, 22
    def polyline(values, color):
        points = " ".join(f"{left + i / max(len(values)-1, 1)*600:.2f},{top + (v-lower)/(upper-lower)*195:.2f}"
                          for i, v in enumerate(values))
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
    ticks = ""
    for value in np.linspace(lower, upper, 5):
        y = top + (value-lower)/(upper-lower)*195
        ticks += f'<path d="M55 {y:.1f}H655" stroke="#e4e8ef"/><text x="45" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>'
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Cp at mid-span, CFD reference and model prediction">'
            f'{ticks}{polyline(t, "#152c47")}{polyline(p, "#e07024")}'
            '<text x="345" y="249" text-anchor="middle">Surface point index (lower + upper surface)</text>'
            f'<text x="57" y="14">Cp · span row {row} · negative upward</text></svg>')


def surface_plot(geometry, field):
    geom, cp = np.asarray(geometry), np.asarray(field)
    if geom.ndim != 3 or geom.shape[0] != 3 or cp.shape != geom.shape[1:]:
        return '<p>Surface coordinates unavailable for this case.</p>'
    # Sample the structured surface, preserving full coordinates in the NPZ artifact.
    stride_s, stride_c = max(1, cp.shape[0] // 25), max(1, cp.shape[1] // 65)
    # CRMpert uses x streamwise, y thickness, z spanwise.
    x, y = geom[0, ::stride_s, ::stride_c], geom[2, ::stride_s, ::stride_c]
    vals = cp[::stride_s, ::stride_c]
    lo, hi = float(cp.min()), float(cp.max())
    dx, dy = max(float(np.ptp(x)), 1e-9), max(float(np.ptp(y)), 1e-9)
    scale = min(570 / dx, 210 / dy)
    circles = []
    for xx, yy, value in zip(x.ravel(), y.ravel(), vals.ravel()):
        f = (float(value)-lo) / max(hi-lo, 1e-9)
        color = f"hsl({235*(1-f):.1f},70%,50%)"
        circles.append(f'<circle cx="{45+(xx-x.min())*scale:.2f}" cy="{235-(yy-y.min())*scale:.2f}" r="2.5" fill="{color}"/>')
    return ('<svg viewBox="0 0 680 270" role="img" aria-label="Model Cp on wing surface, planform projection">'
            + ''.join(circles)
            + f'<text x="45" y="262">Predicted Cp: {lo:.3f} (blue) to {hi:.3f} (red) · x/z projection, both surfaces</text></svg>')


def write_report(report, directory):
    directory = Path(directory)
    if report["mode"] not in ("synthetic_interface_smoke", "real_checkpoint_dataset_evaluation"):
        raise ValueError("Unknown evaluation mode; refusing to label this as a model experiment")
    synthetic = report["mode"] == "synthetic_interface_smoke"
    reference_label = "Synthetic reference" if synthetic else "CFD surface reference"
    prediction_label = "Synthetic perturbation" if synthetic else "Model"
    cards = []
    for case in report["cases"]:
        relative_path = Path(case["array_file"])
        array_path = (directory / relative_path).resolve()
        if not array_path.is_relative_to(directory.resolve()):
            raise ValueError("Array artifact must stay inside report directory")
        with np.load(array_path, allow_pickle=False) as data:
            plot = cp_plot(data["truth"], data["prediction"])
            surface = surface_plot(data["geometry"], data["prediction"][0])
        rows = []
        for key in ("CL", "CD", "CM"):
            metric = case["coefficient_errors"].get(key, {})
            rows.append(f'<tr><td>{key}</td><td>{number(metric.get("reference"))}</td>'
                        f'<td>{number(metric.get("predicted"))}</td><td>{number(metric.get("absolute_error"))}</td></tr>')
        condition = case["condition"]
        check = case.get("integration_consistency_errors") or {}
        integration_note = (f'<p class="small">Reference reintegration discrepancy: CL {number(check.get("CL",{}).get("absolute_error"))}, '
                            f'CD {number(check.get("CD",{}).get("absolute_error"))}, CM {number(check.get("CM",{}).get("absolute_error"))}. '
                            'These differences exist before model prediction; CM reference convention needs care.</p>') if check else ''
        cards.append(f'<section><h2>Sample {html.escape(str(case["sample_id"]))} · Wing {html.escape(str(case["shape_id"]))}</h2>'
                     f'<p>Mach {number(condition["mach"])} · AoA {number(condition["alpha_deg"])}° · '
                     f'Cp RMSE {number(case["field_errors"]["Cp"]["rmse"])} · Prediction {number(case["end_to_end_prediction_s"])} s</p>'
                     f'<div class="plots"><div>{plot}<p class="legend">● {reference_label} &nbsp; <span>● {prediction_label}</span></p></div><div>{surface}</div></div>'
                     f'<table><thead><tr><th>Coefficient</th><th>{reference_label}</th><th>{prediction_label}</th><th>Absolute error</th></tr></thead>'
                     '<tbody>' + ''.join(rows) + '</tbody></table>' + integration_note + '</section>')
    banner = ("SYNTHETIC INTERFACE TEST — no neural model, CFD or speedup evidence" if synthetic else
              "REAL CHECKPOINT + PUBLISHED CFD DATA — speedup requires matched live CFD timing")
    timing = report["prediction_timing"]
    content = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MDO · AeroTransformer evaluation</title>
<style>body{{margin:0;background:#f3f5f8;color:#172c46;font:16px system-ui,sans-serif}}main{{max-width:1160px;margin:auto;padding:40px 24px}}h1{{font-size:34px;margin-bottom:8px}}h2{{font-size:20px}}.banner{{background:#fff0da;border-left:5px solid #e07024;padding:14px;margin:20px 0}}section{{background:white;border:1px solid #dfe5ed;border-radius:10px;padding:24px;margin:24px 0}}.plots{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}svg{{width:100%;font:11px system-ui}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{text-align:left;border-bottom:1px solid #e8edf3;padding:10px}}.legend span{{color:#e07024}}.small{{font-size:13px;color:#526277}}@media(max-width:750px){{.plots{{grid-template-columns:1fr}}}}</style>
<main><p>MDO / REPRODUCTION WORKBENCH</p><h1>Wing prediction → evidence</h1>
<div class="banner">{banner}</div>
<p>{report['case_count']} cases · Median prediction {number(timing['median_s'])} s · Device {html.escape(str(report['device']))}</p>
<p class="small">{html.escape(report.get('reference_source',''))}<br>{html.escape(report.get('split_claim',''))}<br>{html.escape(report.get('timing_scope',''))}</p>
{''.join(cards)}
<p class="small">Pressure and friction are surface predictions. This report does not demonstrate converged aeroelastic MDO or CFD replacement validity outside the evaluated cases. Machine-readable records: evaluation.json and case_*.npz.</p>
<p class="small">Real-data reports use CRMpert by the AeroTransformer authors, CC-BY-SA-4.0: <a href="https://huggingface.co/datasets/thuerey-group/CRMpert">dataset and attribution</a>. Predictions/plots are generated by this workbench.</p></main></html>'''
    path = directory / "report.html"
    path.write_text(content, encoding="utf-8")
    return path
