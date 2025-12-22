# Speculative Decoding: Draft Model Alignment & Acceptance Rate Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This research-grade machine learning project investigates the empirical relationship between draft model alignment and token acceptance rates in **Speculative Decoding** (Leviathan et al., 2023). 

By instrumenting an inference engine utilizing a lightweight draft model (e.g., `distilgpt2`) and a larger target model (e.g., `gpt2-xl`), we study how the Kullback-Leibler (KL) divergence between their output probability distributions correlates with empirical acceptance rates across diverse task domains. Furthermore, we train supervised machine learning models (Logistic Regression, XGBoost, MLPs) to predict token acceptance prior to target-model verification, demonstrating potential compute savings through adaptive decoding.

---

## Setup Instructions

### Prerequisites
- Python 3.8+ (tested on Python 3.11)
- Pip package manager

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Tushar-Jawale/speculative-decoding.git
   cd speculative-decoding
   ```

2. Create and activate a virtual environment:
   - **Windows (CMD):**
     ```cmd
     python -m venv venv
     venv\Scripts\activate.bat
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

The central entry point for the project is `main.py`. You can execute the entire experimental pipeline or configure the speculation window size ($K$) and compute device.

### Running the Full Benchmark
To execute the complete pipeline (inference engine verification, cross-domain empirical study, and acceptance rate prediction):
```bash
python main.py --device auto
```

### K-Sensitivity Analysis
To run an analysis sweeping different speculation window sizes ($K \in \{2, 4, 6, 8\}$):
```bash
python main.py --sweep-k
```

### CLI Arguments
- `--k INT`: Speculation window size / number of draft tokens per step (default: `4`).
- `--device [cuda|cpu|auto]`: Compute device (default: `auto`).
- `--prompts-per-domain INT`: Override number of prompts per domain in the benchmark (default: `50`).
- `--sweep-k`: Run K-sensitivity analysis.

---

## Running Tests

The project includes a comprehensive unit test suite covering KV cache rollback, checkpointing, engine distribution correctness (via Kolmogorov-Smirnov test), and cache sequence consistency.

Run the entire test suite using `pytest`:
```bash
pytest tests/ -v
```

Or run individual test modules:
```bash
pytest tests/test_kv_cache.py -v
pytest tests/test_engine.py -v
```
