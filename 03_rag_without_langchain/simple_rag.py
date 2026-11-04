# Simple RAG without LangChain:
# 1) split text into chunks  2) embed them  3) find top 2 by cosine similarity
# 4) build a prompt for a language model.
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: read the file and make one chunk per line (each line is one short sentence)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents.txt")
with open(path, encoding="utf-8") as f:
    text = f.read()
chunks = [line.strip() for line in text.splitlines() if line.strip()]
print(f"Step 1: split text into {len(chunks)} chunks")

# Step 2: turn every chunk into a vector (downloads the model the first time)
print("Step 2: loading model and embedding chunks...")
model = SentenceTransformer("all-MiniLM-L6-v2")
# normalize_embeddings=True makes each vector length 1,
# so cosine similarity becomes a simple dot product.
chunk_vectors = model.encode(chunks, normalize_embeddings=True)

# Step 3: embed the question and find the 2 most similar chunks
question = "How does a model avoid memorizing the training data?"
print("Step 3: question =", question)
question_vector = model.encode([question], normalize_embeddings=True)[0]
similarities = chunk_vectors @ question_vector      # cosine similarity for each chunk
top_ids = np.argsort(similarities)[::-1][:2]        # indexes of the 2 highest scores
for rank, i in enumerate(top_ids, start=1):
    print(f"  Top {rank} (score {similarities[i]:.3f}): {chunks[i]}")

# Step 4: build the prompt that you would send to a language model
context = "\n".join(f"- {chunks[i]}" for i in top_ids)
prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {question}
Answer:"""
print("\nStep 4: final prompt\n" + "-" * 40)
print(prompt)
