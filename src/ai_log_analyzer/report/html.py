
from pathlib import Path
from typing import List, Dict
import datetime as dt
import html

def render_report(path: str, summary_text: str, top_templates: List[Dict], top_clusters: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.utcnow().isoformat() + "Z"
    def row_tpl(t):
        return f"<tr><td>{html.escape(t['template'])}</td><td>{t['count']}</td><td>{t.get('z', '')}</td></tr>"
    def row_cluster(c):
        ex = html.escape(c['example'])
        return f"<tr><td>{c['cluster']}</td><td>{c['size']}</td><td>{ex}</td></tr>"
    html_txt = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>AI Log Analyzer Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif; margin: 2rem; }}
h1,h2 {{ margin: 0.2rem 0; }}
.card {{ border: 1px solid #eee; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.04);}}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #eee; }}
code, pre {{ background: #f7f7f8; padding: 2px 6px; border-radius: 6px; }}
.badge {{ display: inline-block; background: #eef; padding: 2px 8px; border-radius: 999px; }}
</style>
</head>
<body>
<h1>AI Log Analyzer Report</h1>
<p class="badge">Generated: {now}</p>

<div class="card">
<h2>Executive Summary</h2>
<p>{html.escape(summary_text)}</p>
</div>

<div class="card">
<h2>Top Templates (frequency & z-score)</h2>
<table><thead><tr><th>Template</th><th>Count</th><th>Z</th></tr></thead>
<tbody>
{''.join(row_tpl(t) for t in top_templates[:50])}
</tbody></table>
</div>

<div class="card">
<h2>Top Clusters</h2>
<table><thead><tr><th>Cluster #</th><th>Size</th><th>Example</th></tr></thead>
<tbody>
{''.join(row_cluster(c) for c in top_clusters[:50])}
</tbody></table>
</div>

<footer><small>Local-first • Open-source models: all-MiniLM-L6-v2, flan-t5-small</small></footer>
</body>
</html>
    """.strip()
    Path(path).write_text(html_txt, encoding="utf-8")
    return path
