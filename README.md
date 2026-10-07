## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/adamwetzli/master_thesis
```

### 2. Change into the directory
```bash
cd master_thesis
```

### 2.5 (Optional) Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install packages
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Explore as you wish
```bash
# For example run main
python ./scripts/main.py
```

### Hints
- **Data file missing**: `data/csv_files/forex_master_data/aggregated_complete_data.csv` is omitted due to file size limits

- **How to generate data**:
  - Run `main.py` from the root directory (full pipeline)
  - Or manually call `aggregate_forex_data()` from `./src/utils.py`

- **Running scripts**: Always execute from the project **root** (`/master_thesis/`) since scripts use relative paths
