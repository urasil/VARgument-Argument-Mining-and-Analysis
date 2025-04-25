# VARgument: Football Argument Mining Tool

VARgument is a prototype tool designed to assist football journalists in crafting more consistent and informed arguments by mining and analysing existing arguments from SkySports Premier League articles.

---

## 📂 Repository Structure

```
├── requirements.txt          # Python dependencies
├── RoBERTa_models.txt        # (Not required for running VARgument, argument embeddings are pre-computed in final_knowledgebase.csv)
├── code/                     # Source code folder, includes dataset creation, argument mining, and argument analysis code
│   └── pipeline/main.py      # Entry point for VARgument 
│   └── final_knowledgebase.csv   # Git LFS–tracked knowledge base
└── run.bat                   # Windows batch launcher
```

---

## Prerequisites

- **Python 3.9** or later
- **Git LFS** installed (required for pulling the `.csv` files)

---

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/urasil/VARgument-Argument-Mining-and-Analysis.git
   cd VARgument-Argument-Mining-and-Analysis
   ```

2. **Create and activate a virtual environment** (name must be `venv`):

   - **Windows (PowerShell)**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux**
     ```bash
     Use the equivalent commands for macOS/Linux
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Running VARgument

### Cross‑Platform (macOS/Linux/Windows)

```bash
cd code
python pipeline/main.py
```

This launches the interactive CLI (VARgument tool) for inputting your argument and retrieving similar sentences, sentiment and temporal analysis.

### Windows Quick Start

1. Ensure Python 3.9 and dependencies are installed.
2. Double‑click **run.bat** in the project root.

If the batch file fails, follow the cross‑platform steps above.

---
