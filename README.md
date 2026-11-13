# ml-practice

Small beginner projects for learning machine learning with Python.

```
ml-practice/
├── requirements.txt
├── setup.bat              (Windows setup)
├── setup.sh               (Git Bash / Linux / macOS setup)
├── 01_kmeans_clustering/
│   ├── kmeans_basic.py
│   ├── elbow_method.py
│   ├── silhouette_score.py
│   └── random_init_trap.py
├── 02_ensemble_methods/
│   ├── voting.py
│   ├── stacking.py
│   └── blending.py
├── 03_rag_without_langchain/
│   ├── documents.txt
│   └── simple_rag.py
├── 04_advanced_clustering/
│   ├── kmeans_vs_dbscan.py
│   └── gmm_clustering.py
├── 05_dimensionality_reduction/
│   ├── pca_demo.py
│   └── autoencoder_demo.py
├── 06_anomaly_detection/
│   ├── make_data.py
│   ├── zscore_detection.py
│   ├── isolation_forest.py
│   ├── lof_detection.py
│   └── compare_methods.py
└── 07_association_rules/
    └── apriori_demo.py
```


## Setup (do this once)

You need Python 3.9 or newer. Check with `python --version`.

### Quick way: run the setup script
```bash
# Windows (CMD or PowerShell)
setup.bat

# Git Bash, Linux, macOS, WSL
bash setup.sh
```
The script creates a `.venv` folder and installs everything in `requirements.txt`.

### Manual way (step by step)
1. Open a terminal in the `ml-practice` folder.
2. Create the virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate it (you should see `(.venv)` in the prompt):
   ```bash
   # Windows CMD
   .venv\Scripts\activate
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Git Bash
   source .venv/Scripts/activate
   # Linux, macOS, WSL
   source .venv/bin/activate
   ```
4. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
   `sentence-transformers` is big (it installs PyTorch), so this can take several minutes.
5. Check that it works:
   ```bash
   python -c "import sklearn, matplotlib, sentence_transformers; print('all good')"
   ```

### Every new terminal
Activate the environment again (step 3) before running any file.
To leave it, type `deactivate`.

### Common problems
- **PowerShell says scripts are disabled:** run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
- **`python` not found:** try `py` (Windows) or `python3` (Linux/macOS).
- **No plot window (WSL or server):** run in Windows or add `import matplotlib; matplotlib.use("Agg")`.

## How to run

Every file runs on its own. Go into the folder, then run the file.

### 01_kmeans_clustering
```bash
cd 01_kmeans_clustering
python kmeans_basic.py       # K-means with K=3, plots clusters and centres
python elbow_method.py       # WCSS for K=1..10, marks the elbow
python silhouette_score.py   # silhouette score for K=2..10, prints best K
python random_init_trap.py   # random init vs k-means++ side by side
```

### 02_ensemble_methods (breast cancer dataset)
```bash
cd 02_ensemble_methods
python voting.py     # hard and soft voting with 3 models
python stacking.py   # stacking with logistic regression as meta-model
python blending.py   # blending by hand using a holdout set
```
Each file prints the accuracy of the single models and of the ensemble.

### 03_rag_without_langchain
```bash
cd 03_rag_without_langchain
python simple_rag.py   # chunk -> embed -> top 2 by cosine similarity -> prompt
```
The first run downloads the `all-MiniLM-L6-v2` model (needs internet).
Edit `documents.txt` or the `question` in the script to experiment.

### 04_advanced_clustering
```bash
cd 04_advanced_clustering
python kmeans_vs_dbscan.py   # moon-shaped data: K-means vs DBSCAN side by side
python gmm_clustering.py     # Gaussian Mixture Model, prints group probabilities
```

### 05_dimensionality_reduction (digits dataset)
```bash
cd 05_dimensionality_reduction
python pca_demo.py           # PCA from 64 features to 2, plus explained variance
python autoencoder_demo.py   # PyTorch autoencoder: digits -> 2 numbers -> rebuilt images
```

### 06_anomaly_detection (fake bank transactions)
```bash
cd 06_anomaly_detection
python make_data.py            # makes the fake data (the other files use it)
python zscore_detection.py     # flags payments with Z-score above 3
python isolation_forest.py     # Isolation Forest, anomalies in red
python lof_detection.py        # Local Outlier Factor, anomalies in red
python compare_methods.py      # prints how many anomalies each method found
```

### 07_association_rules
```bash
cd 07_association_rules
python apriori_demo.py   # supermarket bills -> rules with support, confidence, lift
```

## Notes
- Plot windows must be closed before a script finishes.
- No LangChain is used. The RAG code is plain Python, NumPy and sentence-transformers.
