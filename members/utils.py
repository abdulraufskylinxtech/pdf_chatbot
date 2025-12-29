import os
import faiss
import pickle
import re
import fitz  
import numpy as np
from llama_cpp import Llama 
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from django.conf import settings
from PyPDF2 import PdfReader

# model_path = "D:/Python projects/pdf_chatbot/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# llm = Llama(
#     model_path=model_path,
#     n_ctx=4096,     
#     n_threads=6,    
#     n_batch=256
# )

# DOC_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# embed_model = SentenceTransformer(DOC_EMBED_MODEL_NAME)
# EMBED_DIM = embed_model.get_sentence_embedding_dimension()
# print("Embedding dimension:", EMBED_DIM)

FAISS_DIR = os.path.join(settings.BASE_DIR, "faiss_indexes")
os.makedirs(FAISS_DIR, exist_ok=True)



def extract_text_from_pdf_with_fitz(path):
    """
    Extract text using PyMuPDF (fitz).
    If PyMuPDF fails, fallback to PyPDF2.
    Preserves original PDF line structure exactly (no flattening).
    """

    text_parts = []

    try:
        doc = fitz.open(path)
        for page in doc:
            
            page_text = page.get_text("text") or ""
            text_parts.append(page_text.strip())
        doc.close()

    except Exception as e:
        
        try:
            reader = PdfReader(path)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            text = "\n".join(pages)
            return text
        except Exception:
            return f"Error extracting PDF: {e}"

    
    full_text = "\n\n\n\n".join(text_parts)

    return clean_extracted_text_preserve_lines(full_text)


def clean_extracted_text_preserve_lines(text: str) -> str:
    """Cleans and reconstructs PDF-extracted text while preserving paragraph structure."""

    if not text:
        return ""

    text = text.replace("\ufeff", "").replace("\x00", "").replace("\xa0", " ")
    text = re.sub(r'(?i)(copyright|all rights reserved|no reproduction|provided by|licensee|not for resale).*', '', text)
    text = re.sub(r'(?i)ISO\s*\d{3,5}(:\d{4})?', '', text)
    text = re.sub(r'(?i)page\s*\d+', '', text)
    text = re.sub(r'(?i)reference number.*', '', text)
    # text = re.sub(r'(?i)(contents|table of contents|bibliography|foreword|introduction)\s+.*', '', text)
    text = re.sub(r'---+', '', text)
    text = re.sub(r'[•·●▪▶►\-\–\—_,]+', ' ', text)
    text = re.sub(r'[“”"\'`´]', '', text)
    text = re.sub(r'[=+*/\\|]+', ' ', text)
    text = re.sub(r'[`´^¨~]', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[.,;:!?]{2,}', '.', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'NOTE\s*[:\-]?.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(\w)\s*\n\s*(\w)', r'\1 \2', text)
    text = re.sub(r'\b([A-Za-z])\s(?=[a-z])', r'\1', text)
    text = re.sub(r'(?<!\b(be|to|an|in|on|by|at|as|of|or|if|is|it|do|so|no))\b([A-Za-z])\s(?=[a-z])', r'\2', text)
    text = re.sub(r'\b[A-Z]\)', '', text)
    text = re.sub(r'\b\)', '', text)
    text = re.sub(r'\(\s*[A-Za-z0-9]+\s*\)', '', text)
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = []
    buffer = ""

    for line in lines:
        if len(line) < 60 and not line.endswith(('.', ':')):
            buffer += " " + line
        else:
            if buffer:
                joined.append(buffer.strip())
                buffer = ""
            joined.append(line)

    if buffer:
        joined.append(buffer.strip())

    clean_text = "\n".join(joined)
    clean_text = re.sub(r'\n{2,}', '\n', clean_text).strip()

    return clean_text


# def build_faiss_index_from_text(text, index_path, chunk_size=800, overlap=100):
#     """
#     Builds FAISS index + metadata with page and line references.
#     Each chunk will include page_no, line_start, line_end info.
#     """
    
#     meta = []

#     pages = text.split("--- PAGE BREAK ---")
#     chunks = []
#     page_no = 1
#     line_counter = 0

#     for page_text in pages:
#         lines = page_text.strip().splitlines()
#         cleaned_lines = [ln.strip() for ln in lines if ln.strip()]
#         if not cleaned_lines:
#             page_no += 1
#             continue

#         text_on_page = " ".join(cleaned_lines)
#         sentences = re.split(r'(?<=[.!?]) +', text_on_page)

#         current_chunk = ""
#         start_line = 0

#         for i, sentence in enumerate(sentences):
#             if len(current_chunk) + len(sentence) < chunk_size:
#                 if not current_chunk:
#                     start_line = line_counter
#                 current_chunk += sentence + " "
#             else:
#                 chunks.append(current_chunk.strip())
#                 meta.append({
#                     "page_no": page_no,
#                     "line_start": start_line,
#                     "line_end": line_counter,
#                 })
#                 current_chunk = sentence + " "
#                 start_line = line_counter
#             line_counter += 1

#         if current_chunk:
#             chunks.append(current_chunk.strip())
#             meta.append({
#                 "page_no": page_no,
#                 "line_start": start_line,
#                 "line_end": line_counter,
#             })

#         page_no += 1

   
#     if not chunks:
#         raise ValueError("No chunks generated for indexing.")

#     embeddings = embed_model.encode(chunks, convert_to_numpy=True)
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(embeddings, dtype="float32"))
#     faiss.write_index(index, index_path)
 
#     meta_path = index_path.replace(".index", "_meta.pkl")
#     with open(meta_path, "wb") as f:
#         pickle.dump({"chunks": chunks, "meta": meta}, f)

#     print(f" Built FAISS index with {len(chunks)} chunks and metadata: {index_path}")
#     return index_path


def load_faiss_index(index_path):
    """
    Loads FAISS index + metadata (page and line info for each chunk).
    Returns:
        index  – FAISS index object
        chunks – list of text chunks
        meta   – list of dicts with {page_no, line_start, line_end}
    """

    meta_path = index_path.replace(".index", "_meta.pkl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing index or metadata for: {index_path}")
  
    index = faiss.read_index(index_path)
 
    with open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
  
    if isinstance(meta_data, list):
        chunks = meta_data
        meta = [{"page_no": None, "line_start": None, "line_end": None}] * len(chunks)
    else:
        chunks = meta_data.get("chunks", [])
        meta = meta_data.get("meta", [{"page_no": None, "line_start": None, "line_end": None}] * len(chunks))

    print(f" Loaded FAISS index with {len(chunks)} chunks and metadata from {meta_path}")
    return index, chunks, meta


def delete_faiss_index(file_id, user_id):
    base = FAISS_DIR
    idx = os.path.join(base, f"user_{user_id}_file_{file_id}.index")
    meta = idx.replace(".index", "_meta.pkl")
    for p in (idx, meta):
        if os.path.exists(p):
            os.remove(p)


