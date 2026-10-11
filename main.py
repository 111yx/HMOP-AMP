import argparse
from evolutionary_algorithm import EvolutionaryAlgorithm
from membrane_system import MembraneSystem, _hypervolume_2d, _get_pareto_points, HV_REF_POINT
from peptide import Peptide
import sys
import os
import random
import time
import csv
import numpy as np

# Import AttentionNetwork classes and make them available in __main__
# This is needed for PyTorch to correctly load the saved Attention models
try:
    from AMP_classification import AttentionNetwork as ClassificationAttentionNetwork
    from AMP_classification import PositionalEncoding
    from AMP_regression import AttentionNetwork as RegressionAttentionNetwork
    from AMP_regression import PositionalEncoding as RegressionPositionalEncoding
except ImportError:
    # Add current directory to path if import fails
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from AMP_classification import AttentionNetwork as ClassificationAttentionNetwork
    from AMP_classification import PositionalEncoding
    from AMP_regression import AttentionNetwork as RegressionAttentionNetwork
    from AMP_regression import PositionalEncoding as RegressionPositionalEncoding

# Make the AttentionNetwork class available under its original name in __main__
# This is crucial because PyTorch looks for this exact class name when loading models
AttentionNetwork = ClassificationAttentionNetwork

def _heatmap_html_helicity(x_labels, y_labels, matrix, xname, yname, label, colormap="blue"):
    """为螺旋度报告生成 HTML 热图表格（通用版）。"""
    all_vals = [v for row in matrix for v in row if v is not None]
    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0

    def cell_color(v):
        if vmax == vmin:
            ratio = 0.5
        else:
            ratio = (v - vmin) / (vmax - vmin)
        if colormap == "purple":
            r = int(255 - ratio * 120)
            g = int(255 - ratio * 180)
            b = 255
        elif colormap == "green":
            r = int(255 - ratio * 180)
            g = int(255 - ratio * 50)
            b = int(255 - ratio * 180)
        else:  # blue
            r = int(255 - ratio * 180)
            g = int(255 - ratio * 100)
            b = int(255 - ratio * 20)
        return f"rgb({r},{g},{b})"

    header = f"<tr><th>{label} \\u2193{yname} / {xname}\\u2192</th>"
    for x in x_labels:
        header += f"<th>{x}</th>"
    header += "</tr>"

    body = ""
    for i, y in enumerate(y_labels):
        body += f"<tr><th>{y}</th>"
        for j, x in enumerate(x_labels):
            val = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else None
            if val is None:
                body += "<td style='background:#eee;color:#999;'>—</td>"
            else:
                text_color = '#fff' if (vmax-vmin)>0 and (val-vmin)/(vmax-vmin)>0.6 else '#333'
                body += f"<td style='background:{cell_color(val)};color:{text_color}'>{round(val, 4)}</td>"
        body += "</tr>"

    return f"<table class='heat'><thead>{header}</thead><tbody>{body}</tbody></table>"


def _generate_helicity_report(rows, eps_grid, skin_eps_grid, weight_grid,
                               html_path, repeats, eps_start):
    """
    根据螺旋度参数扫描结果生成 HTML 可视化报告。

    三维参数空间：(eps_end, skin_helicity_eps, penalty_weight)
    可视化策略：
      - 热图：固定 weight=中位数，eps_end × skin_eps 热图
      - 折线图：固定 skin_eps/weight=中位数，各指标随 eps_end 变化
      - 原始数据表
    """
    import json
    from collections import defaultdict

    # ── 数据聚合：按 (eps_end, skin_eps, weight) 取均值 ──────────────────
    metric_keys = [
        "m2_feasible_ratio", "m2_best_feasible_mic", "m2_pareto_size",
        "m2_hypervolume", "m2_avg_helicity", "m2_avg_violation",
        "archive_hypervolume", "feasible_hypervolume",
        "archive_size", "archive_pareto_size",
        "archive_best_mic", "archive_best_hemo",
        "archive_avg_mic", "archive_avg_hemo",
        "archive_avg_helicity",
        "elapsed_sec"
    ]

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["eps_end"], r["skin_helicity_eps"], r["penalty_weight"])
        for mk in metric_keys:
            v = r.get(mk)
            if v is not None and str(v) not in ("NA", ""):
                try:
                    agg[key][mk].append(float(v))
                except (ValueError, TypeError):
                    pass

    # 均值
    mean_data = {}
    for key, metrics in agg.items():
        mean_data[key] = {mk: (sum(vs) / len(vs) if vs else None)
                          for mk, vs in metrics.items()}

    # ── 参考固定值（取中位数）─────────────────────────────────────────────
    ref_eps  = eps_grid[len(eps_grid) // 2] if eps_grid else eps_grid[0]
    ref_skin = skin_eps_grid[len(skin_eps_grid) // 2] if skin_eps_grid else skin_eps_grid[0]
    ref_wt   = weight_grid[len(weight_grid) // 2] if weight_grid else weight_grid[0]

    # ── 热图矩阵函数（固定 weight=ref_wt，行=eps_end，列=skin_eps）────────
    def make_hm(metric):
        mat = []
        for e in eps_grid:
            row = []
            for s in skin_eps_grid:
                val = mean_data.get((e, s, ref_wt), {}).get(metric)
                row.append(round(val, 6) if val is not None else None)
            mat.append(row)
        return mat

    hv_mat    = make_hm("archive_hypervolume")
    fhv_mat   = make_hm("feasible_hypervolume")
    fr_mat    = make_hm("m2_feasible_ratio")
    hv2_mat   = make_hm("m2_hypervolume")
    sz_mat    = make_hm("archive_size")
    viol_mat  = make_hm("m2_avg_violation")

    # ── 折线图数据（固定 skin_eps=ref_skin, weight=ref_wt，x=eps_end）─────
    line_eps  = eps_grid
    line_hv   = [mean_data.get((e, ref_skin, ref_wt), {}).get("archive_hypervolume")
                 for e in line_eps]
    line_fhv  = [mean_data.get((e, ref_skin, ref_wt), {}).get("feasible_hypervolume")
                 for e in line_eps]
    line_fr   = [mean_data.get((e, ref_skin, ref_wt), {}).get("m2_feasible_ratio")
                 for e in line_eps]
    line_hv2  = [mean_data.get((e, ref_skin, ref_wt), {}).get("m2_hypervolume")
                 for e in line_eps]
    line_viol = [mean_data.get((e, ref_skin, ref_wt), {}).get("m2_avg_violation")
                 for e in line_eps]

    # ── 折线图数据（固定 eps_end=ref_eps, weight=ref_wt，x=skin_eps）───────
    line_skin = skin_eps_grid
    line_hv_s  = [mean_data.get((ref_eps, s, ref_wt), {}).get("archive_hypervolume")
                  for s in line_skin]
    line_fr_s  = [mean_data.get((ref_eps, s, ref_wt), {}).get("m2_feasible_ratio")
                  for s in line_skin]

    # ── 折线图数据（固定 eps_end=ref_eps, skin_eps=ref_skin，x=weight）────
    line_wt  = weight_grid
    line_hv_w = [mean_data.get((ref_eps, ref_skin, w), {}).get("archive_hypervolume")
                 for w in line_wt]
    line_fr_w = [mean_data.get((ref_eps, ref_skin, w), {}).get("m2_feasible_ratio")
                 for w in line_wt]

    # ── 原始数据表 ─────────────────────────────────────────────────────────
    table_rows_html = ""
    for r in sorted(rows, key=lambda x: (
            x.get("eps_end", 0), x.get("skin_helicity_eps", 0), x.get("penalty_weight", 0))):
        def fmt(v):
            if v is None or v == "NA" or v == "":
                return "—"
            try:
                return f"{float(v):.4f}"
            except (ValueError, TypeError):
                return str(v)

        table_rows_html += f"""
        <tr>
          <td>{r.get('seed', 'NA')}</td>
          <td><b>{r.get('eps_end', '—')}</b></td>
          <td><b>{r.get('skin_helicity_eps', '—')}</b></td>
          <td><b>{r.get('penalty_weight', '—')}</b></td>
          <td class="metric">{fmt(r.get('archive_hypervolume'))}</td>
          <td class="metric">{fmt(r.get('feasible_hypervolume'))}</td>
          <td class="metric">{fmt(r.get('m2_hypervolume'))}</td>
          <td>{fmt(r.get('m2_feasible_ratio'))}</td>
          <td>{fmt(r.get('m2_pareto_size'))}</td>
          <td>{fmt(r.get('m2_avg_helicity'))}</td>
          <td>{fmt(r.get('m2_avg_violation'))}</td>
          <td>{fmt(r.get('archive_size'))}</td>
          <td>{fmt(r.get('archive_pareto_size'))}</td>
          <td>{fmt(r.get('archive_avg_mic'))}</td>
          <td>{fmt(r.get('archive_avg_hemo'))}</td>
          <td>{fmt(r.get('archive_avg_helicity'))}</td>
          <td>{fmt(r.get('elapsed_sec'))}s</td>
        </tr>"""

    # ── 组装 HTML ───────────────────────────────────────────────────────────
    html_content = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>螺旋度约束参数敏感性分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px;
          background: #f5f7fa; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #e67e22; padding-bottom: 8px; }}
  h2 {{ color: #34495e; margin-top: 36px; }}
  .summary {{ background:#fff; border-radius:8px; padding:16px 24px;
              box-shadow:0 2px 6px rgba(0,0,0,.08); margin-bottom:24px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }}
  .card {{ background:#fff; border-radius:8px; padding:16px;
           box-shadow:0 2px 6px rgba(0,0,0,.08); }}
  canvas {{ max-width:100%; }}
  .heatmap-wrap {{ overflow-x:auto; }}
  table.heat {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.heat th,table.heat td {{ border:1px solid #ddd; padding:6px 10px;
                                  text-align:center; }}
  table.heat th {{ background:#e67e22; color:#fff; }}
  table.data {{ border-collapse:collapse; width:100%; font-size:12px;
                margin-top:16px; }}
  table.data th {{ background:#2c3e50; color:#fff; padding:7px 10px; }}
  table.data td {{ border:1px solid #e0e0e0; padding:5px 9px; }}
  table.data tr:nth-child(even) {{ background:#f9f9f9; }}
  .metric {{ font-weight:bold; color:#2980b9; }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:4px;
          font-size:11px; font-weight:bold; }}
  .tag-good  {{ background:#d5f5e3; color:#1e8449; }}
  .tag-warn  {{ background:#fdebd0; color:#d35400; }}
  .tag-bad   {{ background:#fadbd8; color:#922b21; }}
  .legend {{ font-size:12px; color:#666; margin-top:8px; }}
  .note {{ font-size:12px; color:#888; margin-top:4px; }}
</style>
</head>
<body>
<h1>螺旋度约束参数敏感性分析报告</h1>
<div class="summary">
  <b>实验设置</b>：
  eps_start = {eps_start} &nbsp;|&nbsp;
  eps_end ∈ {eps_grid} &nbsp;|&nbsp;
  skin_helicity_eps ∈ {skin_eps_grid} &nbsp;|&nbsp;
  penalty_weight ∈ {weight_grid} &nbsp;|&nbsp;
  重复次数 = {repeats} &nbsp;|&nbsp;
  总运行次数 = {len(rows)}
  <br><br>
  <b>参考固定值（热图/折线图用）</b>：
  weight = {ref_wt}（中位数）&nbsp;|&nbsp;
  skin_eps = {ref_skin}（中位数）&nbsp;|&nbsp;
  eps_end = {ref_eps}（中位数）
  <br><br>
  <b>关键指标说明</b>：
  <span class="tag tag-good">archive_hypervolume</span> 越大越好（精英档案 HV）&nbsp;
  <span class="tag tag-good">feasible_hypervolume</span> 越大越好（可行 HV）&nbsp;
  <span class="tag tag-good">m2_feasible_ratio</span> 越大越好（可行性）&nbsp;
  <span class="tag tag-warn">m2_avg_violation</span> 越小越好（约束违反）
</div>

<h2>📊 热图：精英档案超体积（weight={ref_wt}，均值）</h2>
<div class="card heatmap-wrap">
  {_heatmap_html_helicity(
      [str(s) for s in skin_eps_grid],
      [str(e) for e in eps_grid],
      hv_mat, "skin_eps", "eps_end",
      "archive HV", "blue")}
  <p class="legend">行 = eps_end（螺旋度上限），列 = skin_helicity_eps（皮肤膜约束）；颜色越深 HV 越大</p>
</div>

<h2>📊 热图：可行超体积（weight={ref_wt}，均值）</h2>
<div class="card heatmap-wrap">
  {_heatmap_html_helicity(
      [str(s) for s in skin_eps_grid],
      [str(e) for e in eps_grid],
      fhv_mat, "skin_eps", "eps_end",
      "feasible HV", "green")}
  <p class="legend">颜色越深 HV 越大；None/空缺表示该参数组合无有效数据</p>
</div>

<h2>📊 热图：膜2可行性比率（weight={ref_wt}，均值）</h2>
<div class="card heatmap-wrap">
  {_heatmap_html_helicity(
      [str(s) for s in skin_eps_grid],
      [str(e) for e in eps_grid],
      fr_mat, "skin_eps", "eps_end",
      "m2 feasible ratio", "purple")}
  <p class="legend">颜色越深（紫）可行性越高</p>
</div>

<h2>📊 热图：约束违反均值（weight={ref_wt}，均值）</h2>
<div class="card heatmap-wrap">
  {_heatmap_html_helicity(
      [str(s) for s in skin_eps_grid],
      [str(e) for e in eps_grid],
      viol_mat, "skin_eps", "eps_end",
      "avg violation", "purple")}
  <p class="legend">颜色越深（紫）违反越大（越差）；None 表示无数据</p>
</div>

<h2>📈 折线图：eps_end 影响（固定 skin_eps={ref_skin}, weight={ref_wt}）</h2>
<div class="grid2">
  <div class="card">
    <canvas id="lineHvEps"></canvas>
  </div>
  <div class="card">
    <canvas id="lineFhvEps"></canvas>
  </div>
</div>
<div class="grid2">
  <div class="card">
    <canvas id="lineFrEps"></canvas>
  </div>
  <div class="card">
    <canvas id="lineViolEps"></canvas>
  </div>
</div>

<h2>📈 折线图：skin_helicity_eps 影响（固定 eps_end={ref_eps}, weight={ref_wt}）</h2>
<div class="grid2">
  <div class="card">
    <canvas id="lineHvSkin"></canvas>
  </div>
  <div class="card">
    <canvas id="lineFrSkin"></canvas>
  </div>
</div>

<h2>📈 折线图：penalty_weight 影响（固定 eps_end={ref_eps}, skin_eps={ref_skin}）</h2>
<div class="grid2">
  <div class="card">
    <canvas id="lineHvWt"></canvas>
  </div>
  <div class="card">
    <canvas id="lineFrWt"></canvas>
  </div>
</div>

<h2>📋 原始数据明细</h2>
<div class="card" style="overflow-x:auto;">
<table class="data">
  <thead>
    <tr>
      <th>seed</th>
      <th>eps_end</th><th>skin_eps</th><th>weight</th>
      <th>archive HV</th><th>feasible HV</th><th>m2 HV</th>
      <th>feas_ratio</th><th>m2_pareto</th>
      <th>m2_avg_hel</th><th>m2_avg_viol</th>
      <th>arch_size</th><th>arch_pareto</th>
      <th>avg_mic</th><th>avg_hemo</th><th>avg_helicity</th>
      <th>耗时</th>
    </tr>
  </thead>
  <tbody>{table_rows_html}</tbody>
</table>
</div>

<script>
const eps_labels  = {json.dumps([str(e) for e in eps_grid])};
const skin_labels = {json.dumps([str(s) for s in skin_eps_grid])};
const wt_labels   = {json.dumps([str(w) for w in weight_grid])};

function lineChart(id, labels, datasets, title) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      plugins: {{ title: {{ display: true, text: title, font: {{size:13}} }} }},
      responsive: true,
      scales: {{ y: {{ beginAtZero: false }} }}
    }}
  }});
}}

// eps_end 影响
lineChart('lineHvEps', eps_labels, [{{
  label: 'archive HV',
  data: {json.dumps(line_hv)},
  borderColor: '#3498db', backgroundColor: '#3498db33', tension: 0.3, fill: true
}}], 'archive_hypervolume vs eps_end (skin={ref_skin}, wt={ref_wt})');

lineChart('lineFhvEps', eps_labels, [{{
  label: 'feasible HV',
  data: {json.dumps(line_fhv)},
  borderColor: '#27ae60', backgroundColor: '#27ae6033', tension: 0.3, fill: true
}}], 'feasible_hypervolume vs eps_end');

lineChart('lineFrEps', eps_labels, [{{
  label: 'm2 feasible ratio',
  data: {json.dumps(line_fr)},
  borderColor: '#8e44ad', backgroundColor: '#8e44ad33', tension: 0.3, fill: true
}}], 'm2_feasible_ratio vs eps_end');

lineChart('lineViolEps', eps_labels, [{{
  label: 'm2 avg violation',
  data: {json.dumps(line_viol)},
  borderColor: '#e74c3c', backgroundColor: '#e74c3c33', tension: 0.3, fill: true
}}], 'm2_avg_violation vs eps_end (越小越好)');

// skin_eps 影响
lineChart('lineHvSkin', skin_labels, [{{
  label: 'archive HV',
  data: {json.dumps(line_hv_s)},
  borderColor: '#e67e22', backgroundColor: '#e67e2233', tension: 0.3, fill: true
}}], 'archive_hypervolume vs skin_eps (eps={ref_eps}, wt={ref_wt})');

lineChart('lineFrSkin', skin_labels, [{{
  label: 'm2 feasible ratio',
  data: {json.dumps(line_fr_s)},
  borderColor: '#16a085', backgroundColor: '#16a08533', tension: 0.3, fill: true
}}], 'm2_feasible_ratio vs skin_eps');

// weight 影响
lineChart('lineHvWt', wt_labels, [{{
  label: 'archive HV',
  data: {json.dumps(line_hv_w)},
  borderColor: '#2980b9', backgroundColor: '#2980b933', tension: 0.3, fill: true
}}], 'archive_hypervolume vs penalty_weight (eps={ref_eps}, skin={ref_skin})');

lineChart('lineFrWt', wt_labels, [{{
  label: 'm2 feasible ratio',
  data: {json.dumps(line_fr_w)},
  borderColor: '#8e44ad', backgroundColor: '#8e44ad33', tension: 0.3, fill: true
}}], 'm2_feasible_ratio vs penalty_weight');
</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML 报告已生成：{html_path}")


def _generate_stagnation_report(rows, thresh_grid, window_grid, html_path, repeats):
    """
    根据停滞扫描结果生成 HTML 可视化报告。
    包含：
      1. 参数汇总表
      2. 精英档案 HV 热图（threshold × window）
      3. 通信次数热图（migrate_in_total）
      4. 多样性热图（diversity_m2）
      5. 各指标随参数变化的折线图
    """
    import json

    # ── 汇总：对相同 (thresh, window) 的多次 repeat 取均值 ──────────────
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    metric_keys = [
        "archive_hypervolume", "feasible_hypervolume",
        "archive_size", "archive_pareto_size",
        "migrate_in_total", "migrate_out_total", "lateral_total",
        "stag_membrane1", "stag_membrane2", "stag_membrane3", "stag_skin",
        "diversity_m1", "diversity_m2", "diversity_m3",
        "elapsed_sec"
    ]
    for r in rows:
        key = (r["stagnation_threshold"], r["stagnation_window"])
        for mk in metric_keys:
            v = r.get(mk)
            if v is not None:
                agg[key][mk].append(float(v))

    # 取均值
    mean_data = {}
    for key, metrics in agg.items():
        mean_data[key] = {mk: (sum(vs) / len(vs) if vs else 0.0)
                          for mk, vs in metrics.items()}

    # ── 构造热图矩阵 ──────────────────────────────────────────────────────
    def make_matrix(metric):
        mat = []
        for t in thresh_grid:
            row = []
            for w in window_grid:
                row.append(round(mean_data.get((t, w), {}).get(metric, 0.0), 6))
            mat.append(row)
        return mat

    hv_mat         = make_matrix("archive_hypervolume")
    fhv_mat        = make_matrix("feasible_hypervolume")
    mig_in_mat     = make_matrix("migrate_in_total")
    mig_out_mat    = make_matrix("migrate_out_total")
    div_m2_mat     = make_matrix("diversity_m2")
    archive_sz_mat = make_matrix("archive_size")

    # ── 折线图数据（固定 window=默认，扫描 threshold） ───────────────────
    ref_window = window_grid[len(window_grid) // 2]  # 中间窗口作参考
    line_thresh = thresh_grid
    line_hv       = [mean_data.get((t, ref_window), {}).get("archive_hypervolume", 0) for t in line_thresh]
    line_mig_in   = [mean_data.get((t, ref_window), {}).get("migrate_in_total",   0) for t in line_thresh]
    line_div      = [mean_data.get((t, ref_window), {}).get("diversity_m2",       0) for t in line_thresh]

    ref_thresh = thresh_grid[len(thresh_grid) // 2]
    line_windows  = window_grid
    line_hv_w     = [mean_data.get((ref_thresh, w), {}).get("archive_hypervolume", 0) for w in line_windows]
    line_mig_in_w = [mean_data.get((ref_thresh, w), {}).get("migrate_in_total",   0) for w in line_windows]

    # ── 原始数据表 ────────────────────────────────────────────────────────
    table_rows_html = ""
    for r in sorted(rows, key=lambda x: (x["stagnation_threshold"], x["stagnation_window"])):
        def fmt(v):
            if v is None:
                return "—"
            try:
                return f"{float(v):.4f}" if isinstance(v, float) else str(v)
            except Exception:
                return str(v)

        table_rows_html += f"""
        <tr>
          <td>{r['seed']}</td>
          <td><b>{r['stagnation_threshold']}</b></td>
          <td><b>{r['stagnation_window']}</b></td>
          <td class="metric">{fmt(r.get('archive_hypervolume'))}</td>
          <td class="metric">{fmt(r.get('feasible_hypervolume'))}</td>
          <td>{fmt(r.get('archive_size'))}</td>
          <td>{fmt(r.get('archive_pareto_size'))}</td>
          <td class="comm">{fmt(r.get('migrate_in_total'))}</td>
          <td class="comm">{fmt(r.get('migrate_out_total'))}</td>
          <td class="comm">{fmt(r.get('lateral_total'))}</td>
          <td>{fmt(r.get('diversity_m1'))}</td>
          <td>{fmt(r.get('diversity_m2'))}</td>
          <td>{fmt(r.get('diversity_m3'))}</td>
          <td>{fmt(r.get('elapsed_sec'))}s</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>停滞检测参数敏感性分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px;
          background: #f5f7fa; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; }}
  h2 {{ color: #34495e; margin-top: 36px; }}
  .summary {{ background:#fff; border-radius:8px; padding:16px 24px;
              box-shadow:0 2px 6px rgba(0,0,0,.08); margin-bottom:24px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .card {{ background:#fff; border-radius:8px; padding:16px;
           box-shadow:0 2px 6px rgba(0,0,0,.08); }}
  canvas {{ max-width:100%; }}
  .heatmap-wrap {{ overflow-x:auto; }}
  table.heat {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.heat th,table.heat td {{ border:1px solid #ddd; padding:6px 10px;
                                  text-align:center; }}
  table.heat th {{ background:#3498db; color:#fff; }}
  table.data {{ border-collapse:collapse; width:100%; font-size:12px;
                margin-top:16px; }}
  table.data th {{ background:#2c3e50; color:#fff; padding:7px 10px; }}
  table.data td {{ border:1px solid #e0e0e0; padding:5px 9px; }}
  table.data tr:nth-child(even) {{ background:#f9f9f9; }}
  .metric {{ font-weight:bold; color:#2980b9; }}
  .comm   {{ color:#8e44ad; }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:4px;
          font-size:11px; font-weight:bold; }}
  .tag-good  {{ background:#d5f5e3; color:#1e8449; }}
  .tag-warn  {{ background:#fdebd0; color:#d35400; }}
  .tag-bad   {{ background:#fadbd8; color:#922b21; }}
  .legend {{ font-size:12px; color:#666; margin-top:8px; }}
</style>
</head>
<body>
<h1>停滞检测参数敏感性分析报告</h1>
<div class="summary">
  <b>实验设置</b>：
  stagnation_threshold ∈ {thresh_grid} &nbsp;|&nbsp;
  stagnation_window ∈ {window_grid} &nbsp;|&nbsp;
  重复次数 = {repeats} &nbsp;|&nbsp;
  总运行次数 = {len(rows)}
  <br><br>
  <b>关键指标说明</b>：
  <span class="tag tag-good">archive_hypervolume</span> 越大越好 &nbsp;
  <span class="tag tag-warn">migrate_in_total</span> 过大=通信太频繁 过小=停滞未响应 &nbsp;
  <span class="tag tag-good">diversity_m2</span> 越大越好（多样性）
</div>

<h2>📊 热图：精英档案超体积（archive_hypervolume，均值）</h2>
<div class="card heatmap-wrap">
  {_heatmap_html(thresh_grid, window_grid, hv_mat, "HV")}
  <p class="legend">行 = stagnation_threshold（阈值），列 = stagnation_window（窗口）；颜色越深越优</p>
</div>

<h2>📊 热图：迁入事件次数（migrate_in_total，均值）</h2>
<div class="card heatmap-wrap">
  {_heatmap_html(thresh_grid, window_grid, mig_in_mat, "迁入次数", colormap="purple")}
  <p class="legend">迁入次数过高（阈值过小）说明通信过于频繁；过低（阈值过大或窗口过长）说明通信未发生</p>
</div>

<h2>📊 热图：膜2多样性（diversity_m2，均值）</h2>
<div class="card heatmap-wrap">
  {_heatmap_html(thresh_grid, window_grid, div_m2_mat, "多样性")}
</div>

<h2>📊 热图：精英档案大小（archive_size，均值）</h2>
<div class="card heatmap-wrap">
  {_heatmap_html(thresh_grid, window_grid, archive_sz_mat, "档案大小")}
</div>

<h2>📈 折线图：threshold 影响（固定 window={ref_window}）</h2>
<div class="grid2">
  <div class="card">
    <canvas id="lineHvThresh"></canvas>
  </div>
  <div class="card">
    <canvas id="lineMigThresh"></canvas>
  </div>
</div>

<h2>📈 折线图：window 影响（固定 threshold={ref_thresh}）</h2>
<div class="grid2">
  <div class="card">
    <canvas id="lineHvWindow"></canvas>
  </div>
  <div class="card">
    <canvas id="lineMigWindow"></canvas>
  </div>
</div>

<h2>📋 原始数据明细</h2>
<div class="card" style="overflow-x:auto;">
<table class="data">
  <thead>
    <tr>
      <th>seed</th><th>threshold</th><th>window</th>
      <th>archive HV</th><th>feasible HV</th>
      <th>archive size</th><th>pareto size</th>
      <th>迁入次数</th><th>迁出次数</th><th>横向次数</th>
      <th>div_m1</th><th>div_m2</th><th>div_m3</th>
      <th>耗时</th>
    </tr>
  </thead>
  <tbody>{table_rows_html}</tbody>
</table>
</div>

<script>
const thresh_labels = {json.dumps([str(t) for t in thresh_grid])};
const window_labels = {json.dumps([str(w) for w in window_grid])};

function lineChart(id, labels, datasets, title) {{
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      plugins: {{ title: {{ display: true, text: title, font: {{size:14}} }} }},
      responsive: true,
      scales: {{ y: {{ beginAtZero: false }} }}
    }}
  }});
}}

lineChart('lineHvThresh', thresh_labels, [{{
  label: 'archive_hypervolume (window={ref_window})',
  data: {json.dumps(line_hv)},
  borderColor: '#3498db', backgroundColor: '#3498db33',
  tension: 0.3, fill: true
}}], '不同 threshold 对精英档案 HV 的影响');

lineChart('lineMigThresh', thresh_labels, [{{
  label: 'migrate_in_total (window={ref_window})',
  data: {json.dumps(line_mig_in)},
  borderColor: '#8e44ad', backgroundColor: '#8e44ad33',
  tension: 0.3, fill: true
}}, {{
  label: 'diversity_m2 (window={ref_window})',
  data: {json.dumps(line_div)},
  borderColor: '#27ae60', backgroundColor: '#27ae6033',
  tension: 0.3, yAxisID: 'y1', fill: false
}}], '不同 threshold 对通信次数 & 多样性的影响');

lineChart('lineHvWindow', window_labels, [{{
  label: 'archive_hypervolume (thresh={ref_thresh})',
  data: {json.dumps(line_hv_w)},
  borderColor: '#e67e22', backgroundColor: '#e67e2233',
  tension: 0.3, fill: true
}}], '不同 window 对精英档案 HV 的影响');

lineChart('lineMigWindow', window_labels, [{{
  label: 'migrate_in_total (thresh={ref_thresh})',
  data: {json.dumps(line_mig_in_w)},
  borderColor: '#c0392b', backgroundColor: '#c0392b33',
  tension: 0.3, fill: true
}}], '不同 window 对迁入次数的影响');
</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def _heatmap_html(thresh_grid, window_grid, matrix, label, colormap="blue"):
    """生成 HTML 热图表格，高值深色。"""
    all_vals = [v for row in matrix for v in row if v is not None]
    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0

    def cell_color(v):
        if vmax == vmin:
            ratio = 0.5
        else:
            ratio = (v - vmin) / (vmax - vmin)
        if colormap == "purple":
            r = int(255 - ratio * 120)
            g = int(255 - ratio * 180)
            b = 255
        else:
            r = int(255 - ratio * 180)
            g = int(255 - ratio * 100)
            b = int(255 - ratio * 20)
        return f"rgb({r},{g},{b})"

    header = "<tr><th>" + label + " ↓thresh / window→</th>"
    for w in window_grid:
        header += f"<th>w={w}</th>"
    header += "</tr>"

    body = ""
    for i, t in enumerate(thresh_grid):
        body += f"<tr><th>θ={t}</th>"
        for j, _ in enumerate(window_grid):
            v = matrix[i][j]
            bg = cell_color(v)
            body += f'<td style="background:{bg};">{v:.4f}</td>'
        body += "</tr>"

    return f'<table class="heat"><thead>{header}</thead><tbody>{body}</tbody></table>'


def _generate_ablation_report(rows, ablation_variants, html_path, repeats):
    """
    根据消融实验结果生成 HTML 可视化报告。
    包含：
      1. 实验设置汇总
      2. 各变体核心指标柱状图（archive_hypervolume、feasible_hypervolume、archive_pareto_size）
      3. 通信事件统计（migrate_in、migrate_out、lateral）
      4. 多样性对比（三膜序列多样性）
      5. 运行时间对比
      6. 原始数据明细表
    """
    import json
    from collections import defaultdict

    # ── 去重：基于 (variant, seed) 组合，保留最后一次出现 ─────
    _seen = {}
    for i, r in enumerate(rows):
        key = (r.get("variant", ""), r.get("seed", ""))
        _seen[key] = i  # 后面的覆盖前面的
    rows = [rows[i] for i in sorted(_seen.values())]
    print(f"[报告生成] 去重后保留 {len(rows)} 条记录（原始 {len(rows)} 条）")

    # ── 汇总：对每个变体多次 repeat 取均值 ────────────────────────────────
    metric_keys = [
        "archive_hypervolume", "feasible_hypervolume",
        "archive_size", "archive_pareto_size",
        "migrate_in_total", "migrate_out_total", "lateral_total",
        "diversity_m1", "diversity_m2", "diversity_m3",
        "elapsed_sec"
    ]
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r["variant"]
        for mk in metric_keys:
            val = r.get(mk)
            if val is not None:
                try:
                    agg[v][mk].append(float(val))
                except Exception:
                    pass

    mean_data = {}
    std_data  = {}
    for v, metrics in agg.items():
        mean_data[v] = {}
        std_data[v]  = {}
        for mk, vals in metrics.items():
            mean_data[v][mk] = sum(vals) / len(vals) if vals else 0.0
            if len(vals) > 1:
                m = mean_data[v][mk]
                std_data[v][mk] = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
            else:
                std_data[v][mk] = 0.0

    # ── 颜色方案 ──────────────────────────────────────────────────────────
    variant_colors = {
        'full':             '#27ae60',   # 绿色，标准策略
        'no_comm':          '#e74c3c',
        'fixed_comm':       '#f39c12',
        'no_skin':          '#9b59b6',
        'single_membrane':  '#1abc9c',
        'event_driven':     '#3498db',   # 蓝色，旧版事件驱动（停滞分析内部使用）
    }
    variant_labels = {
        'full':             'Full（标准策略）',
        'no_comm':          'No-Comm（无通信）',
        'fixed_comm':       'Fixed-Comm（固定通信）',
        'no_skin':          'No-Skin（无皮肤膜）',
        'single_membrane':  'Single-Mem（单膜NSGA-II）',
        'event_driven':     'Event-Driven（旧版事件驱动）',
    }

    labels_js    = json.dumps([variant_labels.get(v, v) for v in ablation_variants])
    colors_js    = json.dumps([variant_colors.get(v, '#aaa') for v in ablation_variants])

    def metric_js(mk):
        return json.dumps([round(mean_data.get(v, {}).get(mk, 0.0), 6) for v in ablation_variants])

    def std_js(mk):
        return json.dumps([round(std_data.get(v, {}).get(mk, 0.0), 6) for v in ablation_variants])

    # ── 原始数据表 ────────────────────────────────────────────────────────
    table_rows_html = ""
    for r in sorted(rows, key=lambda x: (ablation_variants.index(x["variant"]) if x["variant"] in ablation_variants else 99, x["seed"])):
        vname = variant_labels.get(r["variant"], r["variant"])
        vcolor = variant_colors.get(r["variant"], "#aaa")
        def fmt(v):
            if v is None:
                return "—"
            try:
                return f"{float(v):.4f}"
            except Exception:
                return str(v)
        table_rows_html += f"""
        <tr>
          <td><span class="vtag" style="background:{vcolor}20;border-left:4px solid {vcolor};padding:2px 8px;">{vname}</span></td>
          <td>{r['seed']}</td>
          <td class="metric">{fmt(r.get('archive_hypervolume'))}</td>
          <td class="metric">{fmt(r.get('feasible_hypervolume'))}</td>
          <td>{fmt(r.get('archive_size'))}</td>
          <td>{fmt(r.get('archive_pareto_size'))}</td>
          <td class="comm">{fmt(r.get('migrate_in_total'))}</td>
          <td class="comm">{fmt(r.get('migrate_out_total'))}</td>
          <td class="comm">{fmt(r.get('lateral_total'))}</td>
          <td>{fmt(r.get('diversity_m1'))}</td>
          <td>{fmt(r.get('diversity_m2'))}</td>
          <td>{fmt(r.get('diversity_m3'))}</td>
          <td>{fmt(r.get('elapsed_sec'))}s</td>
        </tr>"""

    # ── 汇总对比表（均值 ± 标准差）────────────────────────────────────────
    summary_rows_html = ""
    for v in ablation_variants:
        vname  = variant_labels.get(v, v)
        vcolor = variant_colors.get(v, "#aaa")
        def fmt_ms(mk):
            m = mean_data.get(v, {}).get(mk, 0.0)
            s = std_data.get(v, {}).get(mk, 0.0)
            return f"{m:.4f} ± {s:.4f}"
        summary_rows_html += f"""
        <tr>
          <td><span class="vtag" style="background:{vcolor}20;border-left:4px solid {vcolor};padding:2px 8px;">{vname}</span></td>
          <td class="metric">{fmt_ms('archive_hypervolume')}</td>
          <td class="metric">{fmt_ms('feasible_hypervolume')}</td>
          <td>{fmt_ms('archive_pareto_size')}</td>
          <td class="comm">{fmt_ms('migrate_in_total')}</td>
          <td class="comm">{fmt_ms('lateral_total')}</td>
          <td>{fmt_ms('diversity_m2')}</td>
          <td>{fmt_ms('elapsed_sec')}</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>消融实验报告 - 膜计算通信机制</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 24px;
          background: #f0f2f5; color: #333; }}
  h1 {{ color: #1a2d47; border-bottom: 3px solid #3498db; padding-bottom: 10px;
        font-size: 24px; }}
  h2 {{ color: #2c3e50; margin-top: 40px; font-size: 18px; }}
  .summary {{ background:#fff; border-radius:10px; padding:18px 24px;
              box-shadow:0 2px 8px rgba(0,0,0,.10); margin-bottom:28px; line-height:1.8; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }}
  .card {{ background:#fff; border-radius:10px; padding:18px;
           box-shadow:0 2px 8px rgba(0,0,0,.10); }}
  canvas {{ max-width:100%; }}
  table.data {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:8px; }}
  table.data th {{ background:#1a2d47; color:#fff; padding:8px 10px; text-align:left; }}
  table.data td {{ border:1px solid #e0e0e0; padding:6px 10px; }}
  table.data tr:nth-child(even) {{ background:#f8f9fb; }}
  table.sum {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.sum th {{ background:#2c3e50; color:#fff; padding:8px 12px; text-align:left; }}
  table.sum td {{ border:1px solid #dee2e6; padding:8px 12px; }}
  table.sum tr:nth-child(even) {{ background:#f4f6f8; }}
  .metric {{ font-weight:bold; color:#2980b9; }}
  .comm   {{ color:#8e44ad; }}
  .vtag   {{ border-radius:4px; font-size:12px; font-weight:600; }}
  .tag    {{ display:inline-block; padding:3px 10px; border-radius:5px;
             font-size:11px; font-weight:bold; margin:2px; }}
  .tag-full   {{ background:#d6eaf8; color:#1a5276; }}
  .tag-nocomm {{ background:#fadbd8; color:#922b21; }}
  .tag-fixed  {{ background:#fdebd0; color:#784212; }}
  .tag-noskin {{ background:#e8daef; color:#6c3483; }}
  .tag-single {{ background:#d1f2eb; color:#1e8449; }}
  .badge-good {{ background:#d5f5e3; color:#1e8449; padding:2px 8px; border-radius:4px; font-size:11px; }}
  .badge-warn {{ background:#fdebd0; color:#d35400; padding:2px 8px; border-radius:4px; font-size:11px; }}
</style>
</head>
<body>
<h1>🧬 消融实验报告 — 膜计算通信机制</h1>

<div class="summary">
  <b>实验设置</b>：共 {len(ablation_variants)} 个变体，每变体 {repeats} 次重复，总运行 {len(rows)} 次
  <br>
  <b>变体说明</b>：&nbsp;
  <span class="tag tag-full">Full</span> 完整三膜 + 事件驱动通信 + 皮肤膜 &nbsp;
  <span class="tag tag-nocomm">No-Comm</span> 三膜完全隔离无通信 &nbsp;
  <span class="tag tag-fixed">Fixed-Comm</span> 固定间隔强制通信 &nbsp;
  <span class="tag tag-noskin">No-Skin</span> 无皮肤膜精英档案 &nbsp;
  <span class="tag tag-single">Single-Mem</span> 单膜NSGA-II基线 &nbsp;
  <span class="tag" style="background:#d5f5e3;color:#1e8449;">Full-V2</span> 改进通信：去重迁出+自适应迁入+内容感知横向
  <br><br>
  <b>核心指标</b>：
  <span class="badge-good">archive_hypervolume</span> 精英档案超体积（越大越好） &nbsp;
  <span class="badge-good">feasible_hypervolume</span> 可行解超体积（越大越好） &nbsp;
  <span class="badge-good">archive_pareto_size</span> Pareto 前沿大小（越大越好）
</div>

<h2>📊 均值 ± 标准差汇总对比</h2>
<div class="card" style="overflow-x:auto;">
<table class="sum">
  <thead>
    <tr>
      <th>变体</th>
      <th>archive HV ↑</th><th>feasible HV ↑</th>
      <th>Pareto 大小 ↑</th>
      <th>迁入次数</th><th>横向次数</th>
      <th>膜2多样性</th><th>耗时(s)</th>
    </tr>
  </thead>
  <tbody>{summary_rows_html}</tbody>
</table>
</div>

<h2>📈 超体积对比（archive_hypervolume）</h2>
<div class="grid2">
  <div class="card">
    <canvas id="chartHvArchive"></canvas>
  </div>
  <div class="card">
    <canvas id="chartHvFeasible"></canvas>
  </div>
</div>

<h2>📈 档案质量对比</h2>
<div class="grid2">
  <div class="card">
    <canvas id="chartParetoSize"></canvas>
  </div>
  <div class="card">
    <canvas id="chartArchiveSize"></canvas>
  </div>
</div>

<h2>📈 通信事件统计</h2>
<div class="grid3">
  <div class="card">
    <canvas id="chartMigIn"></canvas>
  </div>
  <div class="card">
    <canvas id="chartMigOut"></canvas>
  </div>
  <div class="card">
    <canvas id="chartLateral"></canvas>
  </div>
</div>

<h2>📈 种群多样性对比（不同序列数）</h2>
<div class="grid3">
  <div class="card">
    <canvas id="chartDivM1"></canvas>
  </div>
  <div class="card">
    <canvas id="chartDivM2"></canvas>
  </div>
  <div class="card">
    <canvas id="chartDivM3"></canvas>
  </div>
</div>

<h2>⏱ 运行时间对比</h2>
<div class="card" style="max-width:600px">
  <canvas id="chartTime"></canvas>
</div>

<h2>📋 原始数据明细</h2>
<div class="card" style="overflow-x:auto;">
<table class="data">
  <thead>
    <tr>
      <th>变体</th><th>seed</th>
      <th>archive HV</th><th>feasible HV</th>
      <th>archive size</th><th>pareto size</th>
      <th>迁入次数</th><th>迁出次数</th><th>横向次数</th>
      <th>div_m1</th><th>div_m2</th><th>div_m3</th>
      <th>耗时</th>
    </tr>
  </thead>
  <tbody>{table_rows_html}</tbody>
</table>
</div>

<script>
const labels = {labels_js};
const colors = {colors_js};

function barChart(id, data, errData, title, yLabel) {{
  const ctx = document.getElementById(id);
  if (!ctx) return;
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: yLabel || title,
        data,
        backgroundColor: colors.map(c => c + '99'),
        borderColor: colors,
        borderWidth: 2,
        errorBars: errData ? Object.fromEntries(labels.map((l, i) => [l, {{plus: errData[i], minus: errData[i]}}])) : undefined,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{
        title: {{ display: true, text: title, font: {{size: 13, weight: 'bold'}} }},
        legend: {{ display: false }}
      }},
      scales: {{
        y: {{ beginAtZero: true, title: {{ display: !!yLabel, text: yLabel }} }}
      }}
    }}
  }});
}}

barChart('chartHvArchive',  {metric_js("archive_hypervolume")},  {std_js("archive_hypervolume")},  '精英档案超体积 (archive_hypervolume)', 'HV');
barChart('chartHvFeasible', {metric_js("feasible_hypervolume")}, {std_js("feasible_hypervolume")}, '可行解超体积 (feasible_hypervolume)',   'HV');
barChart('chartParetoSize', {metric_js("archive_pareto_size")},  {std_js("archive_pareto_size")},  'Pareto 前沿大小 (archive_pareto_size)', '个体数');
barChart('chartArchiveSize',{metric_js("archive_size")},         {std_js("archive_size")},          '精英档案大小 (archive_size)',           '个体数');
barChart('chartMigIn',      {metric_js("migrate_in_total")},     {std_js("migrate_in_total")},      '停滞迁入次数 (migrate_in_total)',       '次');
barChart('chartMigOut',     {metric_js("migrate_out_total")},    {std_js("migrate_out_total")},     '精英迁出次数 (migrate_out_total)',      '次');
barChart('chartLateral',    {metric_js("lateral_total")},        {std_js("lateral_total")},         '横向通信次数 (lateral_total)',          '次');
barChart('chartDivM1',      {metric_js("diversity_m1")},         {std_js("diversity_m1")},          '膜1序列多样性 (diversity_m1)',          '唯一序列数');
barChart('chartDivM2',      {metric_js("diversity_m2")},         {std_js("diversity_m2")},          '膜2序列多样性 (diversity_m2)',          '唯一序列数');
barChart('chartDivM3',      {metric_js("diversity_m3")},         {std_js("diversity_m3")},          '膜3序列多样性 (diversity_m3)',          '唯一序列数');
barChart('chartTime',       {metric_js("elapsed_sec")},          {std_js("elapsed_sec")},           '运行时间 (elapsed_sec)',                '秒');
</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


    # ══════════════════════════════════════════════════════════════════════════
    # 增量写入辅助函数（在 main() 外部定义，使其可被模块导入）
    # ══════════════════════════════════════════════════════════════════════════
def _load_completed_seeds(csv_path, id_fields):
    """
    读取已有 CSV，返回已完成的参数组合集合。
    id_fields: 列表，指定用于唯一标识一轮实验的字段名。
    返回: set of tuples
    """
    if not os.path.exists(csv_path):
        return set()
    completed = set()
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = tuple(row.get(f, "").strip() for f in id_fields)
                completed.add(key)
    except Exception:
        pass
    return completed

def _remove_summary_rows(csv_path):
    """从CSV文件中移除所有summary行，保留raw行和header。"""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        raw_rows = [row for row in reader if row.get("run_type") != "summary"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)


def _append_rows(csv_path, fieldnames, rows):
    """追加多条记录到 CSV（自动写 header）。"""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _append_row(csv_path, fieldnames, row):
    """追加一条记录到 CSV（自动写 header）。"""
    _append_rows(csv_path, fieldnames, [row])




def main():
    """主函数"""
    # python main.py --sensitivity-helicity --max-generations 200 --sensitivity-repeats 3   # 默认使用 full 标准策略
    # python main.py --sensitivity-stagnation --max-generations 200 --sensitivity-repeats 3  # 内部强制使用 event_driven 模式
    # python main.py --ablation --max-generations 200 --ablation-repeats 3                   # full 为标准变体
    # 以上均可加 --resume 断点续跑
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="抗菌肽设计 - 多目标多种群进化算法")
    parser.add_argument("--auxiliary-population-size", type=int, default=80, help="辅助种群大小")
    parser.add_argument("--main-population-size", type=int, default=80, help="主种群大小")
    parser.add_argument("--max-generations", type=int, default=200, help="最大进化代数")
    parser.add_argument("--length-range", type=int, nargs=2, default=[5,40], help="肽序列长度范围 (min max)")
    parser.add_argument("--fixed-length", type=int, help="固定长度（如果指定，则生成固定长度的肽序列）")
    parser.add_argument("--helicity-eps-start", type=float, default=0.3, help="helicity 约束的起始 ε")
    parser.add_argument("--helicity-eps-end", type=float, default=0.45, help="helicity 约束的结束 ε（更严格）")
    parser.add_argument("--helicity-eps-fixed", type=float, default=None, help="固定 ε（如果指定则不使用自适应调度）")
    parser.add_argument("--helicity-penalty-weight", type=float, default=0.0, help="helicity 违反度惩罚权重（叠加到 MIC）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（起始种子，实际 seed = seed + rep）")
    parser.add_argument("--seeds", type=str, default="", help="直接指定种子列表（逗号分隔，如 '3,4'），指定后忽略 --seed 和 --*-repeats")
    parser.add_argument("--sensitivity-helicity", action="store_true", help="对螺旋度约束参数做敏感性分析（膜2 ε + 皮肤膜 ε + 惩罚权重）")
    parser.add_argument("--eps-grid", type=str, default="0.45,0.5,0.55,0.6,", help="敏感性分析 ε 取值（逗号分隔，作用于膜2 eps_end）")
    parser.add_argument("--skin-eps-grid", type=str, default="0.2,0.3,0.4", help="敏感性分析皮肤膜螺旋度 ε 取值（逗号分隔）")
    parser.add_argument("--weight-grid", type=str, default="0,5", help="敏感性分析惩罚权重取值（逗号分隔）")
    parser.add_argument("--sensitivity-repeats", type=int, default=1, help="每组参数重复运行次数")
    parser.add_argument("--sensitivity-output", type=str, default="result/experiments/sensitivity_helicity.csv", help="敏感性分析汇总输出 CSV")
    parser.add_argument("--skin-helicity-eps", type=float, default=0.3, help="皮肤膜螺旋度约束 ε（正常运行时使用）")
    # ── 停滞检测敏感性分析 ────────────────────────────────────────────────
    parser.add_argument("--sensitivity-stagnation", action="store_true",
                        help="对停滞检测参数（threshold × window）做敏感性分析")
    parser.add_argument("--stagnation-threshold-grid", type=str, default="0.01,0.03,0.10",
                        help="停滞阈值网格（逗号分隔）")
    parser.add_argument("--stagnation-window-grid", type=str, default="5,10,20",
                        help="停滞窗口网格（逗号分隔）")
    parser.add_argument("--stagnation-output", type=str, default="result/experiments/sensitivity_stagnation.csv",
                        help="停滞分析 CSV 输出路径")
    parser.add_argument("--stagnation-report", type=str, default="result/experiments/sensitivity_stagnation_report.html",
                        help="停滞分析 HTML 可视化报告路径")
    # ── 通信强度敏感性分析 ────────────────────────────────────────────────
    parser.add_argument("--sensitivity-communication", action="store_true",
                        help="对通信强度参数做敏感性分析（迁移规模 × 迁出概率 × 迁入间隔）")
    parser.add_argument("--migration-size-grid", type=str, default="3,5,8,12",
                        help="通信迁移规模网格（逗号分隔）")
    parser.add_argument("--p-out-grid", type=str, default="0.1,0.3,0.5",
                        help="迁出概率网格（逗号分隔）")
    parser.add_argument("--interval-grid", type=str, default="fixed3,linear,sparse",
                        help="迁入间隔策略网格（逗号分隔），可选: fixed3, fixed5, linear, sparse")
    parser.add_argument("--sensitivity-comm-output", type=str,
                        default="result/experiments/sensitivity_communication.csv",
                        help="通信敏感性分析输出 CSV")
    parser.add_argument("--sensitivity-comm-report", type=str,
                        default="result/experiments/sensitivity_communication_report.html",
                        help="通信敏感性分析 HTML 报告")
    # ── 消融实验：膜计算通信机制消融 ─────────────────────────────────────
    parser.add_argument("--ablation", action="store_true",
                        help="运行膜计算通信机制消融实验（full/no_comm/fixed_comm/no_skin/single_membrane）")
    parser.add_argument("--ablation-variants", type=str,
                        default="full,no_comm,fixed_comm,no_skin,single_membrane",
                        help="消融实验变体列表（逗号分隔，full 为标准策略）")
    parser.add_argument("--ablation-repeats", type=int, default=3,
                        help="每个消融变体重复运行次数")
    parser.add_argument("--ablation-output", type=str, default="result/experiments/ablation_results.csv",
                        help="消融实验 CSV 输出路径")
    parser.add_argument("--ablation-report", type=str, default="result/experiments/ablation_report.html",
                        help="消融实验 HTML 可视化报告路径")
    parser.add_argument("--ablation-merge-csv", type=str, default="",
                        help="生成报告时额外合并的 CSV 文件路径（逗号分隔，用于将旧实验结果合入报告）")
    parser.add_argument("--fixed-comm-interval", type=int, default=5,
                        help="fixed_comm 消融变体中强制迁移的间隔代数（默认 5）")
    parser.add_argument("--resume", action="store_true",
                        help="从已有 CSV 续跑（跳过已完成的轮次）")

    args = parser.parse_args()

    # ── 自动创建实验输出目录 ─────────────────────────────────────────────
    _output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "experiments")
    os.makedirs(_output_dir, exist_ok=True)

    print("抗菌肽设计 - 多目标多种群进化算法")
    print("=" * 50)
    # 根据参数决定是否使用固定长度
    if args.fixed_length is not None:
        print(f"使用固定长度: {args.fixed_length}")
        fixed_length = args.fixed_length
        length_range = (args.fixed_length, args.fixed_length)
    else:
        print(f"使用可变长度范围: {args.length_range[0]}-{args.length_range[1]}")
        fixed_length = None
        length_range = tuple(args.length_range)
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    # ======================================================================
    # 实验A：膜计算通信机制消融实验（增量写入 + 断点续跑）
# ======================================================================
    if args.ablation:
        ablation_variants = [v.strip() for v in args.ablation_variants.split(",") if v.strip()]
        repeats = max(1, args.ablation_repeats)

        # single_membrane 变体：膜2大小 = 三膜总和，模拟相同评估预算
        single_membrane_size = (args.auxiliary_population_size + args.main_population_size * 2)

        # ── CSV 字段定义 ──────────────────────────────────────────────
        ablation_fieldnames = [
            "run_type",
            "variant", "seed",
            "archive_hypervolume", "feasible_hypervolume",
            "phase1_final_hv",
            "archive_size", "archive_pareto_size",
            "archive_best_mic", "archive_best_hemo",
            "archive_avg_mic", "archive_avg_hemo",
            "migrate_in_total", "migrate_out_total", "lateral_total",
            "stag_membrane1", "stag_membrane2", "stag_membrane3", "stag_skin",
            "diversity_m1", "diversity_m2", "diversity_m3",
            "elapsed_sec"
        ]
        # 追加聚合统计行所需的 _mean / _std 列
        _ablation_stats_fields = [
            "archive_hypervolume", "feasible_hypervolume", "phase1_final_hv",
            "elapsed_sec", "archive_pareto_size", "migrate_in_total",
            "archive_avg_mic", "archive_avg_hemo"
        ]
        ablation_fieldnames += [f"{f}_mean" for f in _ablation_stats_fields]
        ablation_fieldnames += [f"{f}_std"  for f in _ablation_stats_fields]

        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                args.ablation_output)

        # ── 处理已有 CSV：续跑 vs 覆盖 ──────────────────────────────
        completed_ablation = set()
        rows_ablation = []
        if args.resume and os.path.exists(csv_path):
            # 续跑模式：加载已完成记录，后续追加
            completed_ablation = _load_completed_seeds(csv_path, ["variant", "seed"])
            print(f"[续跑模式] 检测到已有 CSV，已完成 {len(completed_ablation)} 条记录，将跳过")
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows_ablation.append(row)
        elif os.path.exists(csv_path):
            # 非续跑但 CSV 已存在：备份旧文件，从头开始（避免重复追加）
            _backup_path = csv_path + ".bak"
            import shutil
            shutil.move(csv_path, _backup_path)
            print(f"[提示] 已有 CSV 已备份至：{_backup_path}，将创建新文件")

        # ── 确定 seed 列表 ──────────────────────────────────────────────
        if args.seeds.strip():
            seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        else:
            seed_list = [args.seed + rep for rep in range(repeats)]

        total_runs = len(ablation_variants) * len(seed_list)
        run_no = 0
        new_runs = 0

        for run_seed in seed_list:
            for variant in ablation_variants:
                run_no += 1

                # ── 跳过已完成 ──────────────────────────────────────────
                if (variant, str(run_seed)) in completed_ablation:
                    print(f"\n[{run_no}/{total_runs}] [跳过] variant={variant}, seed={run_seed} (已完成)")
                    continue

                print(f"\n[{run_no}/{total_runs}] variant={variant}, seed={run_seed}")
                new_runs += 1

                random.seed(run_seed)
                np.random.seed(run_seed)
                try:
                    import torch
                    torch.manual_seed(run_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(run_seed)
                except Exception:
                    pass

                t0 = time.time()

                # ── 根据变体调整种群大小 ──────────────────────────────
                if variant == 'single_membrane':
                    ea = EvolutionaryAlgorithm(
                        auxiliary_population_size=args.auxiliary_population_size,
                        main_population_size=single_membrane_size,
                        max_generations=args.max_generations,
                        length_range=length_range,
                        fixed_length=fixed_length,
                        helicity_eps_start=args.helicity_eps_start,
                        helicity_eps_end=args.helicity_eps_end,
                        helicity_eps_fixed=args.helicity_eps_fixed,
                        helicity_penalty_weight=args.helicity_penalty_weight,
                        skin_helicity_eps=args.skin_helicity_eps,
                        ablation_mode=variant,
                        fixed_comm_interval=args.fixed_comm_interval
                    )
                else:
                    ea = EvolutionaryAlgorithm(
                        auxiliary_population_size=args.auxiliary_population_size,
                        main_population_size=args.main_population_size,
                        max_generations=args.max_generations,
                        length_range=length_range,
                        fixed_length=fixed_length,
                        helicity_eps_start=args.helicity_eps_start,
                        helicity_eps_end=args.helicity_eps_end,
                        helicity_eps_fixed=args.helicity_eps_fixed,
                        helicity_penalty_weight=args.helicity_penalty_weight,
                        skin_helicity_eps=args.skin_helicity_eps,
                        ablation_mode=variant,
                        fixed_comm_interval=args.fixed_comm_interval
                    )

                ea.run()
                elapsed = time.time() - t0

                ms = ea.membrane_system

                # ── 收集精英档案 / 最终解集 ───────────────────────────
                # 所有变体统一：archive_raw = 精英集合（Pareto排序+拥挤距离裁剪到 max_archive_size）
                # 目的：使 archive_size 指标跨变体可比
                if variant in ('no_skin', 'single_membrane'):
                    if variant == 'single_membrane':
                        all_inds = ms.membrane2.individuals
                    else:
                        all_inds = (ms.membrane1.individuals +
                                    ms.membrane2.individuals +
                                    ms.membrane3.individuals)
                    candidates = [
                        p for p in all_inds
                        if p.antimicrobial_activity is not None
                        and p.hemolytic_activity is not None
                    ]
                    # ── 质量预过滤（避免最终种群快照中混入未收敛差解导致 HV 虚高）──
                    # AMP 预测输出 ∈ [0,1]，值越小抗菌活性越好；
                    # < 0.5 作为"有意义的 AMP 候选"阈值，与完整系统精英档案的螺旋度
                    # 约束过滤效果等价（只保留通过结构验证的个体）。
                    AMP_QUALITY_THRESHOLD = 0.5
                    candidates = [p for p in candidates if p.antimicrobial_activity < AMP_QUALITY_THRESHOLD]
                    # Pareto 排序 + 拥挤距离裁剪（与 _update_elite_archive 逻辑一致）
                    if candidates:
                        fronts = ms.membrane2._non_dominated_sort(candidates)
                        archive_pareto = fronts[0] if fronts else candidates
                        # 逐层填充，裁剪到 max_archive_size
                        cropped = []
                        for fr in fronts:
                            if len(cropped) + len(fr) <= ms.max_archive_size:
                                cropped.extend(fr)
                            else:
                                remaining = ms.max_archive_size - len(cropped)
                                fr_sorted = ms._crowding_sort(fr)
                                cropped.extend(fr_sorted[:remaining])
                                break
                        archive_raw = cropped
                    else:
                        archive_raw    = []
                        archive_pareto = []
                else:
                    archive_raw = [
                        p for p in ms.elite_archive
                        if p.antimicrobial_activity is not None
                        and p.hemolytic_activity is not None
                    ]
                    if archive_raw:
                        fronts = ms.membrane2._non_dominated_sort(archive_raw)
                        archive_pareto = fronts[0] if fronts else archive_raw
                    else:
                        archive_pareto = []

                # ── 计算超体积指标 ─────────────────────────────────────
                if archive_raw:
                    archive_hv = _hypervolume_2d(
                        [(p.antimicrobial_activity, p.hemolytic_activity) for p in archive_raw],
                        HV_REF_POINT
                    )
                    feasible_pts = [
                        p for p in archive_raw
                        if getattr(p, "is_feasible", True)
                        or (p.antimicrobial_activity < 0.5 and p.hemolytic_activity < 0.5)
                    ]
                    feasible_hv = _hypervolume_2d(
                        [(p.antimicrobial_activity, p.hemolytic_activity) for p in feasible_pts],
                        HV_REF_POINT
                    ) if feasible_pts else 0.0
                else:
                    archive_hv  = 0.0
                    feasible_hv = 0.0

                # ── 多样性指标 ────────────────────────────────────────
                def _diversity(pop_inds):
                    return len(set(p.sequence for p in pop_inds)) if pop_inds else 0

                div_m1 = _diversity(ms.membrane1.individuals) if variant != 'single_membrane' else 0
                div_m2 = _diversity(ms.membrane2.individuals)
                div_m3 = _diversity(ms.membrane3.individuals) if variant != 'single_membrane' else 0

                # ── 精英解集平均值 ──────────────────────────────────────
                if archive_raw:
                    avg_mic  = float(np.mean([p.antimicrobial_activity for p in archive_raw]))
                    avg_hemo = float(np.mean([p.hemolytic_activity     for p in archive_raw]))
                else:
                    avg_mic, avg_hemo = 0.0, 0.0

                # ── 立即增量写入 CSV ────────────────────────────────────
                row = {
                    "run_type":             "raw",
                    "variant":              variant,
                    "seed":                 run_seed,
                    "archive_hypervolume":  archive_hv,
                    "feasible_hypervolume": feasible_hv,
                    "phase1_final_hv":      getattr(ms, 'phase1_final_archive_hv', None),
                    "archive_size":         len(archive_raw),
                    "archive_pareto_size":  len(archive_pareto),
                    "archive_best_mic":     min((p.antimicrobial_activity for p in archive_raw), default=None),
                    "archive_best_hemo":    min((p.hemolytic_activity     for p in archive_raw), default=None),
                    "archive_avg_mic":      avg_mic,
                    "archive_avg_hemo":     avg_hemo,
                    "migrate_in_total":     ms.comm_migrate_in_count,
                    "migrate_out_total":    ms.comm_migrate_out_count,
                    "lateral_total":        ms.comm_lateral_count,
                    "stag_membrane1":       ms.stagnation_events.get('exploration', 0),
                    "stag_membrane2":       ms.stagnation_events.get('structure',   0),
                    "stag_membrane3":       ms.stagnation_events.get('safety',      0),
                    "stag_skin":            ms.stagnation_events.get('skin',        0),
                    "diversity_m1":         div_m1,
                    "diversity_m2":         div_m2,
                    "diversity_m3":         div_m3,
                    "elapsed_sec":          round(elapsed, 2)
                }
                _append_row(csv_path, ablation_fieldnames, row)
                rows_ablation.append(row)

                p1hv = getattr(ms, 'phase1_final_archive_hv', None)
                p1hv_str = f"{p1hv:.4f}" if p1hv is not None else "N/A"
                print(f"  archive_hv={archive_hv:.4f}  phase1_hv={p1hv_str}"
                      f"  pareto_size={len(archive_pareto)}"
                      f"  migrate_in={ms.comm_migrate_in_count}  time={elapsed:.1f}s [已写入CSV]")

        print(f"\n消融实验完成：本轮新增 {new_runs} 次运行，合计 {len(rows_ablation)} 条记录")
        print(f"CSV 结果：{csv_path}")

        # ── 追加聚合统计行 ──────────────────────────────────────────────
        raw_rows_ab = [r for r in rows_ablation if r.get("run_type") == "raw"]
        if raw_rows_ab:
            from collections import defaultdict
            groups_ab = defaultdict(list)
            for r in raw_rows_ab:
                groups_ab[r["variant"]].append(r)

            summary_rows_ab = []
            for variant_key, group in groups_ab.items():
                summary_row = {
                    "run_type": "summary",
                    "variant":  variant_key,
                    "seed":      "NA",
                }
                for field in _ablation_stats_fields:
                    vals = [float(r[field]) for r in group if r.get(field) not in ("NA", None, "")]
                    if vals:
                        summary_row[f"{field}_mean"] = round(float(np.mean(vals)), 4)
                        summary_row[f"{field}_std"]  = round(float(np.std(vals)),  4)
                summary_rows_ab.append(summary_row)

            _remove_summary_rows(csv_path)
            _append_rows(csv_path, ablation_fieldnames, summary_rows_ab)
            print(f"已追加 {len(summary_rows_ab)} 条聚合统计行至 CSV")

        # ── 生成 HTML 可视化报告（基于全部已有数据）────────────────────
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 args.ablation_report)

        # ── 合并外部旧 CSV（--ablation-merge-csv）────────────────────────
        merged_rows = list(rows_ablation)  # 本次新运行结果
        if args.ablation_merge_csv.strip():
            for extra_csv in args.ablation_merge_csv.split(","):
                extra_csv = extra_csv.strip()
                if not extra_csv:
                    continue
                extra_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), extra_csv)
                if not os.path.exists(extra_path):
                    print(f"[警告] 合并 CSV 不存在，已跳过：{extra_path}")
                    continue
                with open(extra_path, "r", newline="", encoding="utf-8") as ef:
                    for erow in csv.DictReader(ef):
                        # 只导入 raw 行（跳过 summary 聚合行）
                        if erow.get("run_type", "raw") != "raw":
                            continue
                        # 补全旧行缺少的 phase1_final_hv 字段
                        erow.setdefault("phase1_final_hv", None)
                        merged_rows.append(erow)
                print(f"[合并] 已导入旧 CSV：{extra_path}，当前合并行数：{len(merged_rows)}")

        # 收集本次 + 合并后出现的全部变体（用于报告覆盖范围）
        all_variants_in_data = list(dict.fromkeys(
            [r.get("variant", "") for r in merged_rows
             if r.get("run_type", "raw") == "raw"]
        ))
        report_variants = list(dict.fromkeys(ablation_variants + all_variants_in_data))

        # 将字符串 seed 转回 int（从 CSV 读出的是字符串）
        rows_for_report = []
        for r in merged_rows:
            try:
                r["seed"] = int(r["seed"])
            except Exception:
                pass
            rows_for_report.append(r)
        _generate_ablation_report(rows_for_report, report_variants, html_path, repeats)
        print(f"HTML 报告：{html_path}")
        return

# ======================================================================
# 实验B：停滞检测参数敏感性分析（stagnation_threshold × stagnation_window）
# ======================================================================
    if args.sensitivity_stagnation:
        thresh_grid = [float(x.strip()) for x in args.stagnation_threshold_grid.split(",") if x.strip()]
        window_grid = [int(x.strip())   for x in args.stagnation_window_grid.split(",")    if x.strip()]
        repeats     = max(1, int(args.sensitivity_repeats))

        # ── CSV 字段定义 ──────────────────────────────────────────────
        stag_fieldnames = [
            "run_type",
            "seed", "stagnation_threshold", "stagnation_window",
            "archive_hypervolume", "feasible_hypervolume",
            "archive_size", "archive_pareto_size",
            "archive_best_mic", "archive_best_hemo",
            "archive_avg_mic", "archive_avg_hemo",
            "migrate_in_total", "migrate_out_total", "lateral_total",
            "stag_membrane1", "stag_membrane2", "stag_membrane3", "stag_skin",
            "diversity_m1", "diversity_m2", "diversity_m3",
            "elapsed_sec"
        ]
        # 追加聚合统计行所需的 _mean / _std 列
        _stag_stats_fields = [
            "archive_hypervolume", "feasible_hypervolume", "elapsed_sec",
            "archive_pareto_size", "migrate_in_total",
            "archive_avg_mic", "archive_avg_hemo"
        ]
        stag_fieldnames += [f"{f}_mean" for f in _stag_stats_fields]
        stag_fieldnames += [f"{f}_std"  for f in _stag_stats_fields]

        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                args.stagnation_output)

        # ── 处理已有 CSV：续跑 vs 覆盖 ──────────────────────────────
        completed_stag = set()
        rows_stag = []
        if args.resume and os.path.exists(csv_path):
            completed_stag = _load_completed_seeds(csv_path, ["stagnation_threshold", "stagnation_window", "seed"])
            print(f"[续跑模式] 检测到已有 CSV，已完成 {len(completed_stag)} 条记录，将跳过")
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("run_type") != "summary":
                        rows_stag.append(row)
        elif os.path.exists(csv_path):
            _backup_path = csv_path + ".bak"
            import shutil
            shutil.move(csv_path, _backup_path)
            print(f"[提示] 已有 CSV 已备份至：{_backup_path}，将创建新文件")

        # ── 确定 seed 列表 ──────────────────────────────────────────────
        if args.seeds.strip():
            seed_list_stag = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        else:
            seed_list_stag = [args.seed + rep for rep in range(repeats)]

        total_runs = len(thresh_grid) * len(window_grid) * len(seed_list_stag)
        run_no = 0
        new_runs = 0

        # seed-first 循环顺序：每个 seed 遍历全部参数组合
        for run_seed in seed_list_stag:
            for thresh in thresh_grid:
                for window in window_grid:
                    run_no += 1

                    if (str(thresh), str(window), str(run_seed)) in completed_stag:
                        print(f"[{run_no}/{total_runs}] [跳过] threshold={thresh}, window={window}, seed={run_seed} (已完成)")
                        continue

                    print(f"[{run_no}/{total_runs}] threshold={thresh}, window={window}, seed={run_seed}")
                    new_runs += 1

                    random.seed(run_seed)
                    np.random.seed(run_seed)
                    try:
                        import torch
                        torch.manual_seed(run_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(run_seed)
                    except Exception:
                        pass

                    t0 = time.time()
                    ea = EvolutionaryAlgorithm(
                        auxiliary_population_size=args.auxiliary_population_size,
                        main_population_size=args.main_population_size,
                        max_generations=args.max_generations,
                        length_range=length_range,
                        fixed_length=fixed_length,
                        helicity_eps_start=args.helicity_eps_start,
                        helicity_eps_end=args.helicity_eps_end,
                        helicity_eps_fixed=args.helicity_eps_fixed,
                        helicity_penalty_weight=args.helicity_penalty_weight,
                        skin_helicity_eps=args.skin_helicity_eps,
                        stagnation_threshold=thresh,
                        stagnation_window=window,
                        ablation_mode='event_driven'   # 停滞检测敏感性仅对旧版事件驱动模式有意义
                    )
                    ea.run()
                    elapsed = time.time() - t0

                    ms = ea.membrane_system

                    # ── 精英档案指标 ─────────────────────────────────────
                    archive_raw = [
                        p for p in ms.elite_archive
                        if p.antimicrobial_activity is not None
                        and p.hemolytic_activity is not None
                    ]
                    m2 = ms.membrane2
                    pareto_fronts = m2._non_dominated_sort(archive_raw)
                    archive_pareto = pareto_fronts[0] if pareto_fronts else list(archive_raw)

                    if archive_raw:
                        archive_hv = _hypervolume_2d(
                            [(p.antimicrobial_activity, p.hemolytic_activity) for p in archive_raw],
                            HV_REF_POINT
                        )
                        feasible_pts = [
                            p for p in archive_raw
                            if getattr(p, "is_feasible", True)
                            or (p.antimicrobial_activity < 0.5 and p.hemolytic_activity < 0.5)
                        ]
                        feasible_hv = _hypervolume_2d(
                            [(p.antimicrobial_activity, p.hemolytic_activity) for p in feasible_pts],
                            HV_REF_POINT
                        ) if feasible_pts else 0.0
                    else:
                        archive_hv = 0.0
                        feasible_hv = 0.0

                    best_mic = min([p.antimicrobial_activity for p in archive_raw], default=None) \
                               if archive_raw else None
                    best_hemo = min([p.hemolytic_activity for p in archive_raw], default=None) \
                               if archive_raw else None
                    avg_mic = float(np.mean(
                        [p.antimicrobial_activity for p in archive_raw]
                    )) if archive_raw else 0.0
                    avg_hemo = float(np.mean(
                        [p.hemolytic_activity for p in archive_raw]
                    )) if archive_raw else 0.0

                    # ── 膜多样性 ────────────────────────────────────────
                    def _diversity(pop_inds):
                        return len(set(p.sequence for p in pop_inds)) if pop_inds else 0

                    div_m1 = _diversity(ms.membrane1.individuals)
                    div_m2 = _diversity(ms.membrane2.individuals)
                    div_m3 = _diversity(ms.membrane3.individuals)

                    # ── 通信事件统计 ──────────────────────────────────────
                    migrate_in_total  = ms.comm_migrate_in_count
                    migrate_out_total = ms.comm_migrate_out_count
                    lateral_total     = ms.comm_lateral_count
                    stag_m1 = ms.stagnation_events.get('exploration', 0)
                    stag_m2 = ms.stagnation_events.get('structure',   0)
                    stag_m3 = ms.stagnation_events.get('safety',      0)
                    stag_sk = ms.stagnation_events.get('skin',        0)

                    row = {
                        "run_type":             "raw",
                        "seed":                 run_seed,
                        "stagnation_threshold": thresh,
                        "stagnation_window":    window,
                        "archive_hypervolume":  archive_hv,
                        "feasible_hypervolume": feasible_hv,
                        "archive_size":         len(archive_raw),
                        "archive_pareto_size":  len(archive_pareto),
                        "archive_best_mic":     best_mic,
                        "archive_best_hemo":    best_hemo,
                        "archive_avg_mic":      avg_mic,
                        "archive_avg_hemo":     avg_hemo,
                        "migrate_in_total":     migrate_in_total,
                        "migrate_out_total":    migrate_out_total,
                        "lateral_total":        lateral_total,
                        "stag_membrane1":       stag_m1,
                        "stag_membrane2":       stag_m2,
                        "stag_membrane3":       stag_m3,
                        "stag_skin":            stag_sk,
                        "diversity_m1":         div_m1,
                        "diversity_m2":         div_m2,
                        "diversity_m3":         div_m3,
                        "elapsed_sec":          elapsed
                    }
                    # ── 立即增量写入 CSV ────────────────────────────────
                    _append_row(csv_path, stag_fieldnames, row)
                    rows_stag.append(row)
                    print(f"  HV={archive_hv:.4f}  pareto={len(archive_pareto)}  migrate_in={migrate_in_total}  time={elapsed:.1f}s [已写入CSV]")

        print(f"\n停滞检测敏感性分析完成：本轮新增 {new_runs} 次运行，合计 {len(rows_stag)} 条记录")
        print(f"CSV 结果：{csv_path}")

        # ── 追加聚合统计行 ──────────────────────────────────────────────
        # 只聚合 raw 行（排除已存在的 summary 行）
        raw_rows = [r for r in rows_stag if r.get("run_type") == "raw"]
        if raw_rows:
            from collections import defaultdict
            groups = defaultdict(list)
            for r in raw_rows:
                key = (r["stagnation_threshold"], r["stagnation_window"])
                groups[key].append(r)

            summary_rows = []
            for (thresh, window), group in groups.items():
                summary_row = {
                    "run_type":             "summary",
                    "seed":                 "NA",
                    "stagnation_threshold": thresh,
                    "stagnation_window":    window,
                }
                for field in _stag_stats_fields:
                    vals = [float(r[field]) for r in group if r.get(field) not in ("NA", None, "")]
                    if vals:
                        summary_row[f"{field}_mean"] = round(float(np.mean(vals)), 4)
                        summary_row[f"{field}_std"]  = round(float(np.std(vals)),  4)
                summary_rows.append(summary_row)

            # 用已有的 fieldnames 追加（summary 行字段可能多于 raw，DictWriter 自动处理）
            _remove_summary_rows(csv_path)
            _append_rows(csv_path, stag_fieldnames, summary_rows)
            print(f"已追加 {len(summary_rows)} 条聚合统计行至 CSV")

        # ── 生成 HTML ────────────────────────────────────────────────
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 args.stagnation_report)
        # 数值字段从 CSV 读出的是字符串，需转换；只处理 raw 行
        rows_for_report = []
        for r in rows_stag:
            if r.get("run_type") == "summary":
                continue
            rr = dict(r)
            for k in ["archive_hypervolume", "feasible_hypervolume", "elapsed_sec"]:
                try:
                    rr[k] = float(rr[k])
                except Exception:
                    pass
            for k in ["diversity_m1", "diversity_m2", "diversity_m3",
                       "migrate_in_total", "migrate_out_total", "lateral_total",
                       "stag_membrane1", "stag_membrane2", "stag_membrane3", "stag_skin"]:
                try:
                    rr[k] = int(rr[k])
                except Exception:
                    pass
            rows_for_report.append(rr)
        _generate_stagnation_report(rows_for_report, thresh_grid, window_grid,
                                    html_path, repeats)
        print(f"HTML 报告：{html_path}")
        return

# ======================================================================
# 实验C：螺旋度约束参数敏感性分析
# ======================================================================
    if args.sensitivity_helicity:
        eps_grid = [float(x.strip()) for x in args.eps_grid.split(",") if x.strip() != ""]
        skin_eps_grid = [float(x.strip()) for x in args.skin_eps_grid.split(",") if x.strip() != ""]
        weight_grid = [float(x.strip()) for x in args.weight_grid.split(",") if x.strip() != ""]
        repeats = max(1, int(args.sensitivity_repeats))

        # ── CSV 字段定义 ──────────────────────────────────────────────
        helicity_fieldnames = [
            "run_type",
            "seed",
            "eps_start", "eps_end", "eps_fixed",
            "skin_helicity_eps",
            "penalty_weight",
            "m2_feasible_ratio", "m2_best_feasible_mic", "m2_pareto_size",
            "m2_hypervolume", "m2_avg_helicity", "m2_avg_violation",
            "archive_size", "archive_pareto_size", "archive_hypervolume",
            "feasible_hypervolume",
            "archive_best_mic", "archive_best_hemo",
            "archive_avg_mic", "archive_avg_hemo",
            "archive_avg_helicity",
            "elapsed_sec"
        ]
        # 追加聚合统计行所需的 _mean / _std 列
        _helicity_stats_fields = [
            "archive_hypervolume", "feasible_hypervolume",
            "m2_hypervolume", "m2_feasible_ratio",
            "archive_pareto_size", "elapsed_sec",
            "archive_avg_mic", "archive_avg_hemo"
        ]
        helicity_fieldnames += [f"{f}_mean" for f in _helicity_stats_fields]
        helicity_fieldnames += [f"{f}_std"  for f in _helicity_stats_fields]

        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.sensitivity_output)

        # ── 处理已有 CSV：续跑 vs 覆盖 ──────────────────────────────
        completed_helicity = set()
        rows_helicity = []
        if args.resume and os.path.exists(output_path):
            completed_helicity = _load_completed_seeds(output_path,
                ["eps_end", "skin_helicity_eps", "penalty_weight", "seed"])
            print(f"[续跑模式] 检测到已有 CSV，已完成 {len(completed_helicity)} 条记录，将跳过")
            with open(output_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("run_type") != "summary":
                        rows_helicity.append(row)
        elif os.path.exists(output_path):
            _backup_path = output_path + ".bak"
            import shutil
            shutil.move(output_path, _backup_path)
            print(f"[提示] 已有 CSV 已备份至：{_backup_path}，将创建新文件")

        # ── 确定 seed 列表 ──────────────────────────────────────────────
        if args.seeds.strip():
            seed_list_hel = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        else:
            seed_list_hel = [args.seed + rep for rep in range(repeats)]

        total_runs = (len(eps_grid) * len(skin_eps_grid) * len(weight_grid) * len(seed_list_hel))
        run_counter = [0]
        new_runs = [0]

        # seed-first 循环顺序：每个 seed 遍历全部参数组合
        for run_seed in seed_list_hel:
            for eps_end in eps_grid:
                for skin_eps in skin_eps_grid:
                    for weight in weight_grid:
                        run_counter[0] += 1

                        if (str(eps_end), str(skin_eps), str(weight), str(run_seed)) in completed_helicity:
                            print(f"[{run_counter[0]}/{total_runs}] [跳过] eps_end={eps_end}, "
                                  f"skin_eps={skin_eps}, weight={weight}, seed={run_seed} (已完成)")
                            continue

                        print(f"[{run_counter[0]}/{total_runs}] eps_end={eps_end}, "
                              f"skin_eps={skin_eps}, weight={weight}, seed={run_seed}")
                        new_runs[0] += 1

                        random.seed(run_seed)
                        np.random.seed(run_seed)
                        try:
                            import torch
                            torch.manual_seed(run_seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(run_seed)
                        except Exception:
                            pass

                        start_time = time.time()
                        ea = EvolutionaryAlgorithm(
                            auxiliary_population_size=args.auxiliary_population_size,
                            main_population_size=args.main_population_size,
                            max_generations=args.max_generations,
                            length_range=length_range,
                            fixed_length=fixed_length,
                            helicity_eps_start=args.helicity_eps_start,
                            helicity_eps_end=eps_end,
                            helicity_eps_fixed=args.helicity_eps_fixed,
                            helicity_penalty_weight=weight,
                            skin_helicity_eps=skin_eps
                        )
                        ea.run()
                        elapsed = time.time() - start_time

                        # ── 评估精英档案 ─────────────────────────────────
                        membrane2 = ea.membrane_system.membrane2
                        archive = ea.membrane_system.elite_archive
                        archive = [p for p in archive
                                   if p.antimicrobial_activity is not None
                                   and p.hemolytic_activity is not None]

                        pareto_fronts = membrane2._non_dominated_sort(archive)
                        archive_pareto = pareto_fronts[0] if pareto_fronts else list(archive)

                        if archive:
                            archive_hv = _hypervolume_2d(
                                [(p.antimicrobial_activity, p.hemolytic_activity) for p in archive],
                                HV_REF_POINT
                            )
                            feasible_archive = [
                                p for p in archive
                                if getattr(p, "is_feasible", True)
                                or (p.antimicrobial_activity < 0.5 and p.hemolytic_activity < 0.5)
                            ]
                            feasible_hv = _hypervolume_2d(
                                [(p.antimicrobial_activity, p.hemolytic_activity)
                                 for p in feasible_archive],
                                HV_REF_POINT
                            ) if feasible_archive else 0.0
                        else:
                            archive_hv = 0.0
                            feasible_hv = 0.0

                        best_mic_archive = min(
                            [p.antimicrobial_activity for p in archive], default=None) if archive else None
                        best_hemo_archive = min(
                            [p.hemolytic_activity for p in archive], default=None) if archive else None
                        avg_mic_archive = float(np.mean(
                            [p.antimicrobial_activity for p in archive]
                        )) if archive else 0.0
                        avg_hemo_archive = float(np.mean(
                            [p.hemolytic_activity for p in archive]
                        )) if archive else 0.0
                        avg_helicity_archive = float(np.mean(
                            [getattr(p, "helicity", 0.0) for p in archive]
                        )) if archive else 0.0

                        # ── 评估主种群（膜2）────────────────────────────
                        ea._evaluate_population(membrane2, ea.max_generations - 1)
                        m2_individuals = membrane2.individuals
                        m2_feasible = [p for p in m2_individuals
                                       if getattr(p, "is_feasible", True)]
                        m2_pareto = membrane2.get_pareto_front()

                        if m2_pareto:
                            f1s_m2 = [p.antimicrobial_activity for p in m2_pareto
                                      if p.antimicrobial_activity is not None]
                            f2s_m2 = [p.hemolytic_activity for p in m2_pareto
                                      if p.hemolytic_activity is not None]
                            if f1s_m2 and f2s_m2:
                                m2_hv = _hypervolume_2d(
                                    [(p.antimicrobial_activity, p.hemolytic_activity)
                                     for p in m2_pareto],
                                    HV_REF_POINT
                                )
                            else:
                                m2_hv = 0.0
                        else:
                            m2_hv = 0.0

                        best_feasible_mic = min(
                            [p.antimicrobial_activity for p in m2_feasible
                             if p.antimicrobial_activity is not None], default=None)
                        feasible_ratio = len(m2_feasible) / len(m2_individuals) \
                            if m2_individuals else 0.0
                        avg_helicity_m2 = float(np.mean(
                            [getattr(p, "helicity", 0.0) for p in m2_individuals]
                        )) if m2_individuals else 0.0
                        avg_violation = float(np.mean(
                            [getattr(p, "constraint_violation", 0.0) for p in m2_individuals]
                        )) if m2_individuals else 0.0

                        row = {
                            "run_type":   "raw",
                            "seed":       run_seed,
                            "eps_start":  float(args.helicity_eps_start),
                            "eps_end": float(eps_end),
                            "eps_fixed": args.helicity_eps_fixed,
                            "skin_helicity_eps": float(skin_eps),
                            "penalty_weight": float(weight),
                            "m2_feasible_ratio": feasible_ratio,
                            "m2_best_feasible_mic": best_feasible_mic,
                            "m2_pareto_size": len(m2_pareto),
                            "m2_hypervolume": m2_hv,
                            "m2_avg_helicity": avg_helicity_m2,
                            "m2_avg_violation": avg_violation,
                            "archive_size": len(archive),
                            "archive_pareto_size": len(archive_pareto),
                            "archive_hypervolume": archive_hv,
                            "feasible_hypervolume": feasible_hv,
                            "archive_best_mic": best_mic_archive,
                            "archive_best_hemo": best_hemo_archive,
                            "archive_avg_mic":  avg_mic_archive,
                            "archive_avg_hemo": avg_hemo_archive,
                            "archive_avg_helicity": avg_helicity_archive,
                            "elapsed_sec": elapsed
                        }
                        # ── 立即增量写入 CSV ─────────────────────────────
                        _append_row(output_path, helicity_fieldnames, row)
                        rows_helicity.append(row)
                        print(f"  archive_hv={archive_hv:.4f}  pareto={len(archive_pareto)}"
                              f"  feasible_ratio={feasible_ratio:.2f}  time={elapsed:.1f}s [已写入CSV]")

        print(f"\n螺旋度敏感性分析完成：本轮新增 {new_runs[0]} 次运行，合计 {len(rows_helicity)} 条记录")
        print(f"结果已保存到: {output_path}")

        # ── 追加聚合统计行 ──────────────────────────────────────────────
        raw_rows = [r for r in rows_helicity if r.get("run_type") == "raw"]
        if raw_rows:
            from collections import defaultdict
            groups = defaultdict(list)
            for r in raw_rows:
                key = (r["eps_end"], r["skin_helicity_eps"], r["penalty_weight"])
                groups[key].append(r)

            summary_rows = []
            for (eps_end, skin_eps, weight), group in groups.items():
                summary_row = {
                    "run_type":           "summary",
                    "seed":               "NA",
                    "eps_start":          float(args.helicity_eps_start),
                    "eps_end":            float(eps_end),
                    "eps_fixed":          args.helicity_eps_fixed,
                    "skin_helicity_eps":  float(skin_eps),
                    "penalty_weight":    float(weight),
                }
                for field in _helicity_stats_fields:
                    vals = [float(r[field]) for r in group if r.get(field) not in ("NA", None, "")]
                    if vals:
                        summary_row[f"{field}_mean"] = round(float(np.mean(vals)), 4)
                        summary_row[f"{field}_std"]  = round(float(np.std(vals)),  4)
                summary_rows.append(summary_row)

            _remove_summary_rows(output_path)
            _append_rows(output_path, helicity_fieldnames, summary_rows)
            print(f"已追加 {len(summary_rows)} 条聚合统计行至 CSV")

        # ── 生成 HTML 报告 ─────────────────────────────────────────────────
        html_path = output_path.replace(".csv", ".html")
        _generate_helicity_report(
            rows=rows_helicity,
            eps_grid=eps_grid,
            skin_eps_grid=skin_eps_grid,
            weight_grid=weight_grid,
            html_path=html_path,
            repeats=repeats,
            eps_start=float(args.helicity_eps_start)
        )
        return

    # ======================================================================
    # 通信强度敏感性分析
    # ======================================================================
    if args.sensitivity_communication:
        repeats = max(1, args.sensitivity_repeats)
        interval_strategies = {
            "fixed3":  (3, 3),
            "fixed5":  (5, 5),
            "sparse":  (10, 10),
            "linear":  (3, 10),
        }

        interval_raw = [s.strip() for s in args.interval_grid.split(",") if s.strip() != ""]
        interval_pairs = []
        for s in interval_raw:
            if s in interval_strategies:
                interval_pairs.append(interval_strategies[s])
            else:
                print(f"[警告] 未知间隔策略: {s}，已跳过。可选: {list(interval_strategies.keys())}")

        if not interval_pairs:
            interval_pairs = [(3, 10)]

        mig_grid = [int(x.strip()) for x in args.migration_size_grid.split(",") if x.strip() != ""]
        pout_grid = [float(x.strip()) for x in args.p_out_grid.split(",") if x.strip() != ""]

        print("=" * 60)
        print("通信强度敏感性分析")
        print(f"  迁移规模网格:     {mig_grid}")
        print(f"  迁出概率网格:     {pout_grid}")
        print(f"  迁入间隔策略:     {interval_pairs}")
        print(f"  每组重复次数:     {repeats}")
        print(f"  总参数组合:       {len(mig_grid)} × {len(pout_grid)} × {len(interval_pairs)} = {len(mig_grid) * len(pout_grid) * len(interval_pairs)}")
        print(f"  总运行次数:       {len(mig_grid) * len(pout_grid) * len(interval_pairs) * repeats}")
        print("=" * 60)

        comm_fieldnames = [
            "run_type",
            "seed",
            "base_migration_size", "base_p_out",
            "interval_min", "interval_max",
            "archive_size", "archive_pareto_size", "archive_hypervolume",
            "feasible_hypervolume",
            "archive_best_mic", "archive_best_hemo",
            "archive_avg_mic", "archive_avg_hemo",
            "migrate_in_total", "migrate_out_total", "lateral_total",
            "stag_membrane1", "stag_membrane2", "stag_membrane3", "stag_skin",
            "diversity_m1", "diversity_m2", "diversity_m3",
            "elapsed_sec"
        ]
        _comm_stats_fields = [
            "archive_hypervolume", "feasible_hypervolume",
            "archive_pareto_size", "elapsed_sec",
            "migrate_in_total", "archive_avg_mic", "archive_avg_hemo"
        ]
        comm_fieldnames += [f"{f}_mean" for f in _comm_stats_fields]
        comm_fieldnames += [f"{f}_std"  for f in _comm_stats_fields]

        comm_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        args.sensitivity_comm_output)

        completed_comm = set()
        rows_comm = []
        if args.resume and os.path.exists(comm_output_path):
            completed_comm = _load_completed_seeds(comm_output_path,
                ["base_migration_size", "base_p_out", "interval_min", "interval_max", "seed"])
            print(f"[续跑模式] 检测到已有 CSV，已完成 {len(completed_comm)} 条记录，将跳过")
            with open(comm_output_path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("run_type") != "summary":
                        rows_comm.append(row)
        elif os.path.exists(comm_output_path):
            import shutil
            _backup_path = comm_output_path + ".bak"
            shutil.move(comm_output_path, _backup_path)
            print(f"[提示] 已有 CSV 已备份至：{_backup_path}，将创建新文件")

        if args.seeds.strip():
            seed_list_comm = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        else:
            seed_list_comm = [args.seed + rep for rep in range(repeats)]

        total_runs_comm = (len(mig_grid) * len(pout_grid) * len(interval_pairs) * len(seed_list_comm))
        run_counter_comm = [0]
        new_runs_comm = [0]

        for run_seed in seed_list_comm:
            for mig_size in mig_grid:
                for p_out in pout_grid:
                    for (iv_min, iv_max) in interval_pairs:
                        run_counter_comm[0] += 1

                        key = (str(mig_size), str(p_out), str(iv_min), str(iv_max), str(run_seed))
                        if key in completed_comm:
                            print(f"[{run_counter_comm[0]}/{total_runs_comm}] [跳过] "
                                  f"mig={mig_size}, p_out={p_out}, iv=({iv_min},{iv_max}), seed={run_seed} (已完成)")
                            continue

                        print(f"[{run_counter_comm[0]}/{total_runs_comm}] "
                              f"mig={mig_size}, p_out={p_out}, iv=({iv_min},{iv_max}), seed={run_seed}")
                        new_runs_comm[0] += 1

                        random.seed(run_seed)
                        np.random.seed(run_seed)
                        try:
                            import torch
                            torch.manual_seed(run_seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(run_seed)
                        except Exception:
                            pass

                        start_time = time.time()
                        ea = EvolutionaryAlgorithm(
                            auxiliary_population_size=args.auxiliary_population_size,
                            main_population_size=args.main_population_size,
                            max_generations=args.max_generations,
                            length_range=length_range,
                            fixed_length=fixed_length,
                            helicity_eps_start=args.helicity_eps_start,
                            helicity_eps_end=args.helicity_eps_end,
                            helicity_eps_fixed=args.helicity_eps_fixed,
                            helicity_penalty_weight=args.helicity_penalty_weight,
                            skin_helicity_eps=args.skin_helicity_eps,
                            base_migration_size=mig_size,
                            base_p_out=p_out,
                            interval_min=iv_min,
                            interval_max=iv_max
                        )
                        ea.run()
                        elapsed = time.time() - start_time

                        ms = ea.membrane_system

                        archive_raw = [
                            p for p in ms.elite_archive
                            if p.antimicrobial_activity is not None
                            and p.hemolytic_activity is not None
                        ]
                        if archive_raw:
                            fronts = ms.membrane2._non_dominated_sort(archive_raw)
                            archive_pareto = fronts[0] if fronts else archive_raw
                        else:
                            archive_pareto = []

                        if archive_raw:
                            archive_hv = _hypervolume_2d(
                                [(p.antimicrobial_activity, p.hemolytic_activity) for p in archive_raw],
                                HV_REF_POINT
                            )
                            feasible_pts = [
                                p for p in archive_raw
                                if getattr(p, "is_feasible", True)
                                or (p.antimicrobial_activity < 0.5 and p.hemolytic_activity < 0.5)
                            ]
                            feasible_hv = _hypervolume_2d(
                                [(p.antimicrobial_activity, p.hemolytic_activity) for p in feasible_pts],
                                HV_REF_POINT
                            ) if feasible_pts else 0.0
                        else:
                            archive_hv = 0.0
                            feasible_hv = 0.0

                        def _diversity_comm(pop_inds):
                            return len(set(p.sequence for p in pop_inds)) if pop_inds else 0

                        div_m1 = _diversity_comm(ms.membrane1.individuals)
                        div_m2 = _diversity_comm(ms.membrane2.individuals)
                        div_m3 = _diversity_comm(ms.membrane3.individuals)

                        if archive_raw:
                            avg_mic  = float(np.mean([p.antimicrobial_activity for p in archive_raw]))
                            avg_hemo = float(np.mean([p.hemolytic_activity     for p in archive_raw]))
                        else:
                            avg_mic, avg_hemo = 0.0, 0.0

                        row = {
                            "run_type":             "raw",
                            "seed":                 run_seed,
                            "base_migration_size":  mig_size,
                            "base_p_out":           p_out,
                            "interval_min":         iv_min,
                            "interval_max":         iv_max,
                            "archive_size":         len(archive_raw),
                            "archive_pareto_size":  len(archive_pareto),
                            "archive_hypervolume":  archive_hv,
                            "feasible_hypervolume": feasible_hv,
                            "archive_best_mic":     min((p.antimicrobial_activity for p in archive_raw), default=None),
                            "archive_best_hemo":    min((p.hemolytic_activity     for p in archive_raw), default=None),
                            "archive_avg_mic":      avg_mic,
                            "archive_avg_hemo":     avg_hemo,
                            "migrate_in_total":     ms.comm_migrate_in_count,
                            "migrate_out_total":    ms.comm_migrate_out_count,
                            "lateral_total":        ms.comm_lateral_count,
                            "stag_membrane1":       ms.stagnation_events.get('exploration', 0),
                            "stag_membrane2":       ms.stagnation_events.get('structure', 0),
                            "stag_membrane3":       ms.stagnation_events.get('safety', 0),
                            "stag_skin":            ms.stagnation_events.get('skin', 0),
                            "diversity_m1":         div_m1,
                            "diversity_m2":         div_m2,
                            "diversity_m3":         div_m3,
                            "elapsed_sec":          elapsed
                        }
                        _append_row(comm_output_path, comm_fieldnames, row)
                        rows_comm.append(row)
                        print(f"  archive_hv={archive_hv:.4f}  pareto={len(archive_pareto)}"
                              f"  mig_in={ms.comm_migrate_in_count}  time={elapsed:.1f}s [已写入CSV]")

        print(f"\n通信强度敏感性分析完成：本轮新增 {new_runs_comm[0]} 次运行，"
              f"合计 {len(rows_comm)} 条记录")
        print(f"结果已保存到: {comm_output_path}")

        raw_rows_comm = [r for r in rows_comm if r.get("run_type") == "raw"]
        if raw_rows_comm:
            from collections import defaultdict
            groups = defaultdict(list)
            for r in raw_rows_comm:
                key = (r["base_migration_size"], r["base_p_out"],
                       r["interval_min"], r["interval_max"])
                groups[key].append(r)

            summary_rows_comm = []
            for (mig, pout, iv_min, iv_max), group in groups.items():
                srow = {
                    "run_type": "summary",
                    "seed": "NA",
                    "base_migration_size": int(mig),
                    "base_p_out": float(pout),
                    "interval_min": int(iv_min),
                    "interval_max": int(iv_max),
                }
                for field in _comm_stats_fields:
                    vals = [float(r[field]) for r in group if r.get(field) not in ("NA", None, "")]
                    if vals:
                        srow[f"{field}_mean"] = round(float(np.mean(vals)), 4)
                        srow[f"{field}_std"]  = round(float(np.std(vals)),  4)
                summary_rows_comm.append(srow)

            # 先清除CSV中已有的summary行，避免续跑时重复追加
            _remove_summary_rows(comm_output_path)
            _append_rows(comm_output_path, comm_fieldnames, summary_rows_comm)
            print(f"已追加 {len(summary_rows_comm)} 条聚合统计行至 CSV")

        return

    # ======================================================================
    # 默认运行模式（无实验参数）
    # ======================================================================
    ea = EvolutionaryAlgorithm(
        auxiliary_population_size=args.auxiliary_population_size,
        main_population_size=args.main_population_size,
        max_generations=args.max_generations,
        length_range=length_range,
        fixed_length=fixed_length,
        helicity_eps_start=args.helicity_eps_start,
        helicity_eps_end=args.helicity_eps_end,
        helicity_eps_fixed=args.helicity_eps_fixed,
        helicity_penalty_weight=args.helicity_penalty_weight,
        skin_helicity_eps=args.skin_helicity_eps
    )
    ea.run()
    
    # 获取最好的解决方案
    print("\\\\n寻找最优解决方案...")
    # 注意：get_best_solutions方法可能不存在，我们需要从种群中获取帕累托前沿
    ea._evaluate_population(ea.main_population, ea.max_generations - 1)
    best_solutions = ea.main_population.get_pareto_front()
    
    print("\\\\n最好的10个抗菌肽设计方案:")
    print("-" * 50)
    for i, peptide in enumerate(best_solutions[:10], 1):  # 只显示前10个
        print(f"{i}. 序列: {peptide.sequence}")
        print(f"   长度: {len(peptide.sequence)}")
        if peptide.antimicrobial_activity is not None and peptide.hemolytic_activity is not None:
            print(f"   抗菌活性: {peptide.antimicrobial_activity:.4f}")
            print(f"   溶血活性: {peptide.hemolytic_activity:.4f}")
        else:
            print("   目标值尚未计算")
        print()

    # 显示主种群的一些统计信息
    print("主种群统计信息:")
    print("-" * 30)
    # 计算主种群中满足约束条件的个体比例
    constrained_count = 0
    total_antimicrobial = 0.0
    total_hemolytic = 0.0
    count_with_values = 0
    
    for peptide in ea.main_population.individuals:
        if peptide.antimicrobial_activity is not None and peptide.hemolytic_activity is not None:
            total_antimicrobial += peptide.antimicrobial_activity
            total_hemolytic += peptide.hemolytic_activity
            count_with_values += 1
            
        # 检查是否满足约束条件
        from physicochemical_constraints import PhysicochemicalConstraints
        if PhysicochemicalConstraints.check_constraints(peptide):
            constrained_count += 1
    
    if count_with_values > 0:
        avg_antimicrobial = total_antimicrobial / count_with_values
        avg_hemolytic = total_hemolytic / count_with_values
        print(f"平均抗菌活性: {avg_antimicrobial:.4f}")
        print(f"平均溶血活性: {avg_hemolytic:.4f}")
    
    print(f"满足约束条件的个体比例: {constrained_count}/{len(ea.main_population.individuals)}")
    
    print("\\n程序运行完成!")
    print("\\n生成的帕累托前沿图:")
    # 列出生成的帕累托前沿图
    import glob
    pareto_files = glob.glob("pareto_front_generation_*.png")
    for file in sorted(pareto_files):
        print(f"  - {file}")

if __name__ == "__main__":
    main()
