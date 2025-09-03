
# AI Log Analyzer (Open-Source, Local-First)

A lightweight, **open-source AI-powered log analyzer** you can run locally. It ingests log files (or stdin), parses common formats, detects anomalies, clusters events, and produces human-friendly summaries using **open-source models**. No proprietary APIs.

## ✨ Features
- **Ingestion**: file paths, recursive directories, or `stdin`.
- **Parsing**: timestamp/level extraction + simple templating (numbers/hex/IPs → wildcards).
- **Embeddings**: semantic grouping via `sentence-transformers/all-MiniLM-L6-v2`.
- **Anomaly detection**:
  - Frequency-based z-score on templates
  - Unsupervised **Autoencoder** for vector anomalies
- **Summarization**: local **FLAN-T5** (`google/flan-t5-small`) to explain spikes & errors.
- **Clustering**: mini-batch KMeans to group similar messages.
- **HTML Report** with top anomalies, clusters, and summaries.
- **CLI** via `typer` + pretty output via `rich`.

## 📦 Dependencies

### Core Dependencies
The project relies on the following Python libraries (listed in `requirements.txt`):
- **typer** (0.12.3) - CLI framework
- **rich** (13.7.1) - Enhanced console output
- **pandas** (2.2.2) - Data manipulation and analysis
- **numpy** (1.26.4) - Numerical computing
- **scikit-learn** (1.5.1) - Machine learning algorithms
- **torch** (>=2.2.0) - Deep learning framework
- **transformers** (4.43.4) - NLP models and tokenizers
- **sentence-transformers** (3.0.1) - Sentence embeddings
- **matplotlib** (3.8.4) - Plotting and visualization
- **reportlab** (4.2.0) - PDF report generation
- **tensorflow** (2.16.1) - Machine learning framework

### Additional Libraries Used in Code
The codebase also utilizes these libraries for specific functionalities:
- **click** - Command-line interface creation (dependency of typer)

### Standard Library Usage
Extensively uses Python's standard library including `os`, `re`, `pathlib`, `typing`, `dataclasses`, `datetime`, and others for core functionality.

### Notes
- All libraries used in the code are now properly listed in `requirements.txt`.
- The project is designed to work with open-source models only, ensuring privacy and local execution.

## 🚀 Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Analyze a log file and create a report
python -m ai_log_analyzer.cli analyze --path ./examples/sample.log --report out/report.html

# Read from stdin
cat ./examples/sample.log | python -m ai_log_analyzer.cli analyze --stdin --report out/report.html

# Train autoencoder on your 'normal' logs, then score
python -m ai_log_analyzer.cli train --path ./examples/sample.log --model out/autoencoder.pt
python -m ai_log_analyzer.cli analyze --path ./examples/sample_anomalies.log --autoencoder out/autoencoder.pt --report out/report.html
```

> First run downloads open-source models from Hugging Face:
> - `sentence-transformers/all-MiniLM-L6-v2`
> - `google/flan-t5-small`

License: **Apache-2.0**
