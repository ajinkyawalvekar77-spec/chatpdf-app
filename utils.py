from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)

    text_chunks = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            sentences = text.split("\n")
            text_chunks.extend(sentences)

    embeddings = model.encode(text_chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, text_chunks


def search(index, text_chunks, query):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k=3)

    results = []
    for i in indices[0]:
        results.append(text_chunks[i])

    return "\n".join(results)
