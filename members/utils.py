import string
import os
import json
import csv
import re
import fitz  
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from django.conf import settings


EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

STOPWORDS = set(stopwords.words("english"))





def delete_faiss_index(file_id, user_id):
    """Remove FAISS index and metadata when a file is deleted."""
    base_path = os.path.join(settings.BASE_DIR, "faiss_indexes")
    index_file = os.path.join(base_path, f"user_{user_id}_{file_id}.index")
    meta_file = index_file.replace(".index", "_meta.pkl")

    for path in [index_file, meta_file]:
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted FAISS index: {path}")
            


def clean_text_for_indexing(text):
    # Remove BOMs, invisible characters
    text = text.replace("\ufeff", "").replace("\x00", "")
    
    # Replace newlines and tabs with space
    text = re.sub(r'[\r\n\t]+', ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove emojis and non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Optional: remove bullets, dashes, special symbols
    text = re.sub(r'[•\-\–\—]+', ' ', text)

    # Remove extra punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text.strip()


def extract_file_content(file_path):
    text = ""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        elif ext == ".csv":
            with open(file_path, newline='', encoding="utf-8") as f:
                reader = csv.reader(f)
                text = " ".join([" ".join(row) for row in reader])
        elif ext == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            text += " ".join(f"{k}: {v}" for k, v in entry.items()) + ". "
                        else:
                            text += str(entry) + ". "
                elif isinstance(data, dict):
                    text = " ".join(f"{k}: {v}" for k, v in data.items())
                else:
                    text = str(data)
        else:
            text = "[Unsupported file type]"
    except Exception as e:
        text = f"[Error reading file: {e}]"


    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [w.lower() for w in tokens if w.isalnum() and w.lower() not in stopwords]
    return " ".join(filtered)



def preprocess_sentences(sentences: list) -> list:
    """
    Optionally remove stopwords in a lightweight manner from sentences for embeddings.
    We keep original sentences (for display) and return embeddings based on cleaned sentences.
    """

    cleaned = []
    for s in sentences:
        tokens = [w.lower() for w in word_tokenize(s) if w.isalnum()]
        tokens = [t for t in tokens if t not in STOPWORDS]
        cleaned.append(" ".join(tokens) if tokens else s)
    return cleaned

def make_sentence_embeddings(sentences: list) -> np.ndarray:
    """
    Returns sentence embeddings (np.ndarray shape [n, d])
    """
    if not sentences:
        return np.array([])
 
    cleaned = preprocess_sentences(sentences)
    embs = EMBED_MODEL.encode(cleaned, show_progress_bar=False, convert_to_numpy=True)
    return embs

def find_best_sentence_answer(user_question: str, sentences: list, sentence_embeddings: np.ndarray, threshold: float = 0.55):
    """
    Returns (best_sentence, score) or (None, score) if below threshold.
    threshold recommended 0.55-0.6
    """
    if not sentences or sentence_embeddings.size == 0:
        return None, 0.0
    q_emb = EMBED_MODEL.encode([user_question], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, sentence_embeddings)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score < threshold:
        return None, best_score
    return sentences[best_idx], best_score


