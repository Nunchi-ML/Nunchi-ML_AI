import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/expressway/segment_vds_aadt_v1.csv')
df = df[df['aadt'] > 0].copy().reset_index(drop=True)
y_count = df['glare_front_count'].values
groups = df['route'].values

gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(df))
for tr, te in gkf.split(df, y_count, groups):
    glm = PoissonRegressor(alpha=1.0)
    glm.fit(df.iloc[tr][['annual_glare_hours_fwd']], y_count[tr])
    oof[te] = glm.predict(df.iloc[te][['annual_glare_hours_fwd']])

df['oof'] = oof
df['pred_bin'] = pd.qcut(df['oof'], q=5, duplicates='drop')

bins_data = []
for idx, (name, grp) in enumerate(df.groupby('pred_bin', observed=True)):
    n = len(grp)
    pred_mean = grp['oof'].mean()
    obs_sum = int(grp['glare_front_count'].sum())
    obs_mean = grp['glare_front_count'].mean()
    
    ci_low_count = 0.5 * stats.chi2.ppf(0.025, 2 * obs_sum) if obs_sum > 0 else 0.0
    ci_high_count = 0.5 * stats.chi2.ppf(0.975, 2 * (obs_sum + 1))
    ci_low_mean = ci_low_count / n
    ci_high_mean = ci_high_count / n
    
    bins_data.append({
        'bin': idx + 1,
        'label': f"{idx+1}분위",
        'range_str': str(name),
        'n': n,
        'pred_mean': pred_mean,
        'obs_mean': obs_mean,
        'ci_low': ci_low_mean,
        'ci_high': ci_high_mean,
        'obs_sum': obs_sum,
        'ratio': obs_mean / pred_mean
    })

bdf = pd.DataFrame(bins_data)

# HTML / SVG Calibration Chart 생성
html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>A-GLM 5-Fold OOF 5분위수 캘리브레이션 다이어그램</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
  .card {{ max-width: 860px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }}
  h2 {{ margin-top: 0; font-size: 20px; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
  p.desc {{ font-size: 14px; color: #94a3b8; line-height: 1.6; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
  th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid #334155; }}
  th {{ background: #0f172a; color: #cbd5e1; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .badge-warn {{ background: rgba(239, 68, 68, 0.2); color: #f87171; padding: 2px 6px; border-radius: 4px; font-size: 11px; }}
  .badge-ok {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; padding: 2px 6px; border-radius: 4px; font-size: 11px; }}
  .badge-sparse {{ background: rgba(234, 179, 8, 0.2); color: #facc15; padding: 2px 6px; border-radius: 4px; font-size: 11px; }}
  .svg-container {{ margin-top: 24px; background: #0b1120; border-radius: 8px; padding: 16px; border: 1px solid #334155; }}
</style>
</head>
<body>
<div class="card">
  <h2>📊 A-GLM 5-Fold OOF 국소 캘리브레이션 검증도 (Predicted vs Observed)</h2>
  <p class="desc">
    단변량 Poisson GLM의 5-Fold 교차검증 예측값(x축)과 실제 구간당 사고 발생률(y축)을 비교한 도표입니다.
    회색 점선은 완벽한 캘리브레이션(\(y = x\))을 나타내며, 수직 막대는 <strong>Exact Poisson 95% 신뢰구간</strong>입니다.
  </p>

  <div class="svg-container">
    <svg viewBox="0 0 700 420" width="100%" height="auto">
      <!-- Grid & Axis -->
      <line x1="80" y1="350" x2="660" y2="350" stroke="#475569" stroke-width="1.5" />
      <line x1="80" y1="350" x2="80" y2="30" stroke="#475569" stroke-width="1.5" />
      
      <!-- Axis Labels -->
      <text x="370" y="395" fill="#94a3b8" font-size="12" text-anchor="middle">A-GLM 5-Fold OOF 예측 사고건수 (평균, 건/km)</text>
      <text x="25" y="190" fill="#94a3b8" font-size="12" text-anchor="middle" transform="rotate(-90 25 190)">실제 관측 사고건수 (건/km)</text>
      
      <!-- Ideal 45-degree Line (Scaled: (0,0) -> (80,350), (0.06, 0.06) -> (640, 50)) -->
      <!-- x_scale: x -> 80 + (x / 0.06) * 560 -->
      <!-- y_scale: y -> 350 - (y / 0.06) * 300 -->
      <line x1="80" y1="350" x2="640" y2="50" stroke="#64748b" stroke-width="1.5" stroke-dasharray="6,4" />
      <text x="645" y="45" fill="#64748b" font-size="11">이상선 (y = x)</text>
"""

# 좌표 변환 함수
def to_svg(x, y):
    sx = 80 + (x / 0.06) * 560
    sy = 350 - (y / 0.06) * 300
    return sx, sy

# 5분위 데이터 그리기
for r in bins_data:
    px, py = to_svg(r['pred_mean'], r['obs_mean'])
    _, py_low = to_svg(r['pred_mean'], r['ci_low'])
    _, py_high = to_svg(r['pred_mean'], r['ci_high'])
    
    color = "#f87171" if r['bin'] == 3 else ("#facc15" if r['bin'] in [1, 2] else "#38bdf8")
    
    html_content += f"""
      <!-- {r['label']} Error Bar -->
      <line x1="{px}" y1="{py_low}" x2="{px}" y2="{py_high}" stroke="{color}" stroke-width="2" />
      <line x1="{px-5}" y1="{py_low}" x2="{px+5}" y2="{py_low}" stroke="{color}" stroke-width="2" />
      <line x1="{px-5}" y1="{py_high}" x2="{px+5}" y2="{py_high}" stroke="{color}" stroke-width="2" />
      <!-- Point -->
      <circle cx="{px}" cy="{py}" r="5" fill="{color}" />
      <text x="{px+8}" y="{py+4}" fill="#f8fafc" font-size="11">{r['label']} ({r['ratio']:.2f}x)</text>
"""

html_content += """
    </svg>
  </div>

  <table>
    <thead>
      <tr>
        <th>분위</th>
        <th>구간수 (N)</th>
        <th>평균 예측 (건/km)</th>
        <th>실제 관측 (건/km)</th>
        <th>실제 사고수</th>
        <th>관측/예측 비율</th>
        <th>95% 신뢰구간 (비율)</th>
        <th>상태</th>
      </tr>
    </thead>
    <tbody>
"""

for r in bins_data:
    badge = '<span class="badge-warn">유의미한 과소예측</span>' if r['bin'] == 3 else ('<span class="badge-sparse">표본 희소 (CI 광범위)</span>' if r['bin'] in [1, 2] else '<span class="badge-ok">양호 적합</span>')
    ci_r_low = r['ci_low'] / r['pred_mean']
    ci_r_high = r['ci_high'] / r['pred_mean']
    html_content += f"""
      <tr>
        <td><strong>{r['label']}</strong></td>
        <td>{r['n']:,}개</td>
        <td>{r['pred_mean']:.5f}</td>
        <td>{r['obs_mean']:.5f}</td>
        <td>{r['obs_sum']}건</td>
        <td><strong>{r['ratio']:.2f}배</strong></td>
        <td>[{ci_r_low:.2f}, {ci_r_high:.2f}]</td>
        <td>{badge}</td>
      </tr>
"""

html_content += """
    </tbody>
  </table>
</div>
</body>
</html>
"""

with open('docs/model1_calibration_chart.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Calibration diagram written to docs/model1_calibration_chart.html successfully!")
