import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF INFO logs

import re
import time
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Optional: transformers
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    _TRANSFORMERS_OK = True
except Exception:
    _TRANSFORMERS_OK = False
    AutoTokenizer = AutoModel = torch = None

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

console = Console()

# -----------------------------
# Helpers
# -----------------------------
def ensure_parent_dir(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Text preprocessing
# -----------------------------
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
NUM_RE = re.compile(r"\b\d+\b")
WS_RE = re.compile(r"\s+")

def templateize(msg: str) -> str:
    msg = UUID_RE.sub("<UUID>", msg)
    msg = IP_RE.sub("<IP>", msg)
    msg = HEX_RE.sub("<HEX>", msg)
    msg = NUM_RE.sub("<NUM>", msg)
    return msg

def normalize_line(s: str) -> str:
    s = s.replace("\r", " ").replace("\t", " ")
    s = WS_RE.sub(" ", s).strip()
    return s

def preprocess_logs(raw_lines):
    return [normalize_line(templateize(ln)) for ln in raw_lines if ln.strip()]

# -----------------------------
# Model builders
# -----------------------------
def build_autoencoder(input_dim: int, hidden=[64, 32, 64]):
    model = keras.Sequential([keras.layers.Input(shape=(input_dim,))])
    for h in hidden:
        model.add(keras.layers.Dense(h, activation="relu"))
    model.add(keras.layers.Dense(input_dim, activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse")
    return model

# -----------------------------
# Reports
# -----------------------------
def save_html_report(report_data, output_path: str):
    ensure_parent_dir(output_path)
    df = pd.DataFrame(report_data)
    html_txt = df.to_html(index=False, classes="table table-striped")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'><title>Log Analysis Report</title></head>"
                f"<body><h1>Log Analysis Report</h1>{html_txt}</body></html>")
    console.print(f"[green]HTML report saved to {output_path}[/green]")

def save_pdf_report(report_data, output_path: str):
    ensure_parent_dir(output_path)
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, height - 50, "Log Analysis Report")
    c.setFont("Helvetica", 10)
    y = height - 80
    for row in report_data:
        c.drawString(30, y, f"{row['log']} [{row['anomaly']}]")
        y -= 15
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 50
    c.save()
    console.print(f"[green]PDF report saved to {output_path}[/green]")

def save_pie_chart(report_data, output_path: str):
    ensure_parent_dir(output_path)
    df = pd.DataFrame(report_data)
    counts = df['anomaly'].value_counts()
    labels = counts.index.tolist()
    colors_map = {"No": "green", "Yes": "red"}
    colors = [colors_map.get(lbl, "gray") for lbl in labels]
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors)
    plt.title("Anomaly Distribution")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Pie chart saved to {output_path}[/green]")

# -----------------------------
# Transformer embeddings
# -----------------------------
def get_transformer_model(model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
    if not _TRANSFORMERS_OK:
        raise RuntimeError("Transformers not installed")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if device:
        model.to(device)
    return tokenizer, model

def compute_embeddings(logs, tokenizer, model, device=None, batch_size=16):
    if not logs:
        return np.zeros((0, 384), dtype=np.float32)
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(logs), batch_size):
            batch = logs[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
            if device:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_emb)
    return np.vstack(embeddings)

# -----------------------------
# Read logs
# -----------------------------
def read_logs(path: str, live: bool = False, sleep_sec: float = 1.0):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    if p.is_file():
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            if live:
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        yield line.strip("\n")
                    else:
                        time.sleep(sleep_sec)
            else:
                for line in f:
                    yield line.strip("\n")
    elif p.is_dir():
        for fp in p.rglob("*"):
            if fp.is_file():
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        yield line.strip("\n")

# -----------------------------
# Processing
# -----------------------------
def process_and_report(raw_lines, method, tokenizer=None, transformer_model=None, device=None):
    logs = preprocess_logs(raw_lines)
    if not logs:
        return []

    def make_vectorizer():
        return TfidfVectorizer(max_features=1000, token_pattern=r"(?u)\b\w+\b", lowercase=False, min_df=1)

    report_rows = []

    # Isolation
    if method in ["isolation", "hybrid"]:
        vec = make_vectorizer()
        X = vec.fit_transform(logs).toarray()
        iso_model = IsolationForest(n_estimators=150, contamination=0.1, random_state=42)
        iso_preds = iso_model.fit_predict(X)
        iso_anoms = np.array(["Yes" if p == -1 else "No" for p in iso_preds])
    else:
        iso_anoms = np.array(["No"] * len(logs))

    # Autoencoder
    if method in ["autoencoder", "hybrid"]:
        vec = make_vectorizer()
        X_ae = vec.fit_transform(logs).toarray()
        Xs = StandardScaler().fit_transform(X_ae)
        ae_model = build_autoencoder(Xs.shape[1])
        ae_model.fit(Xs, Xs, epochs=5, batch_size=16, verbose=0)  # fast incremental training
        recon = ae_model.predict(Xs, verbose=0)
        errors = np.mean((Xs - recon) ** 2, axis=1)
        thr = np.percentile(errors, 90)
        ae_anoms = np.array(["Yes" if e > thr else "No" for e in errors])
    else:
        ae_anoms = np.array(["No"] * len(logs))

    # Transformer
    if method in ["transformer", "hybrid"] and tokenizer and transformer_model:
        try:
            emb = compute_embeddings(logs, tokenizer, transformer_model, device=device)
            iso_t = IsolationForest(n_estimators=150, contamination=0.1, random_state=42)
            t_preds = iso_t.fit_predict(emb)
            t_anoms = np.array(["Yes" if p == -1 else "No" for p in t_preds])
        except Exception as e:
            console.print(f"[yellow]Transformer failed: {e}[/yellow]")
            t_anoms = np.array(["No"] * len(logs))
    else:
        t_anoms = np.array(["No"] * len(logs))

    # Combine votes
    if method == "isolation":
        final = iso_anoms
    elif method == "autoencoder":
        final = ae_anoms
    elif method == "transformer":
        final = t_anoms
    else:
        votes = (iso_anoms == "Yes").astype(int) + (ae_anoms == "Yes").astype(int) + (t_anoms == "Yes").astype(int)
        final = np.where(votes >= 2, "Yes", "No")

    for ln, flag in zip(logs, final):
        report_rows.append({"log": ln, "anomaly": flag})
        if flag == "Yes":
            console.print(f"[red][ANOMALY][/red] {ln}")

    return report_rows

# -----------------------------
# CLI
# -----------------------------
@click.group()
def app():
    """AI-Powered Hybrid Log Analyzer CLI."""
    pass

@app.command()
@click.option("--path", required=True, type=str, help="Path to log file or directory")
@click.option("--report", default="out/report.html", type=str, help="HTML report")
@click.option("--pdf", type=str, help="PDF report")
@click.option("--chart", type=str, help="Pie chart")
@click.option("--method", type=click.Choice(["isolation", "autoencoder", "transformer", "hybrid"]), default="hybrid")
@click.option("--device", type=str, default=None, help="Device for transformer (cpu/cuda)")
@click.option("--live", is_flag=True, help="Enable live monitoring")
def analyze(path, report, pdf, chart, method, device, live):
    ensure_parent_dir(report)
    if pdf: ensure_parent_dir(pdf)
    if chart: ensure_parent_dir(chart)

    console.print(f"[blue]Analyzing logs from {path} using {method} (live={live})[/blue]")

    tokenizer = transformer_model = None
    if method in ["transformer", "hybrid"] and _TRANSFORMERS_OK:
        tokenizer, transformer_model = get_transformer_model(device=device)

    buffer_lines = []
    report_rows_total = []
    CHUNK = 1 if live else 10_000
    REPORT_INTERVAL = 50

    for raw in read_logs(path, live=live):
        buffer_lines.append(raw)
        if len(buffer_lines) >= CHUNK:
            report_rows = process_and_report(buffer_lines, method, tokenizer, transformer_model, device)
            buffer_lines = []
            report_rows_total.extend(report_rows)

            if live and len(report_rows_total) >= REPORT_INTERVAL:
                save_html_report(report_rows_total, report)
                if pdf: save_pdf_report(report_rows_total, pdf)
                if chart: save_pie_chart(report_rows_total, chart)
                report_rows_total = []

    if buffer_lines:
        report_rows = process_and_report(buffer_lines, method, tokenizer, transformer_model, device)
        report_rows_total.extend(report_rows)

    # Save final reports
    if report_rows_total:
        save_html_report(report_rows_total, report)
        if pdf: save_pdf_report(report_rows_total, pdf)
        if chart: save_pie_chart(report_rows_total, chart)

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    app()
