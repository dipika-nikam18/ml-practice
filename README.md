# ml-practice

Small beginner projects for learning machine learning with Python.

```
ml-practice/
├── requirements.txt
├── 01_kmeans_clustering/
│   ├── kmeans_basic.py
│   ├── elbow_method.py
│   ├── silhouette_score.py
│   └── random_init_trap.py
├── 02_ensemble_methods/
│   ├── voting.py
│   ├── stacking.py
│   └── blending.py
└── 03_rag_without_langchain/
    ├── documents.txt
    └── simple_rag.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

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

## Notes
- Plot windows must be closed before a script finishes.
- No LangChain is used. The RAG code is plain Python, NumPy and sentence-transformers.
