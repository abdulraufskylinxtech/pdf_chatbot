import os, json, fitz, nltk
import pandas as pd
import requests
import PyPDF2
import docx
import uuid
from django.utils import timezone
import faiss
import traceback
import pickle
from django.conf import settings
import numpy as np
from PyPDF2 import PdfReader
import re
import redis
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize,PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters
from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth import login as auth_login, logout,get_user_model
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.core.cache import cache
from django.http import JsonResponse,HttpResponseBadRequest
from django.contrib import messages
from llama_cpp import Llama
from .forms import SignUpForm, LoginForm
from .models import UploadedFile, ChatbotQA,FileChunk,ApiUser,ApiChatMessage,ApiUploadedFile,InterviewSession,InterviewQuestion,InterviewResponse


nltk.data.path.append(r'D:\Python projects\pdf_chatbot\pdf_chat\nltk_data')
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, download_dir=r"D:\Python projects\pdf_chatbot\pdf_chat\nltk_data")
nltk.download('stopwords')

punkt_param = PunktParameters()
tokenizer = PunktSentenceTokenizer(punkt_param)

LLM_MODEL = Llama(
    model_path="D:/Python projects/pdf_chatbot/models/phi-3-mini-4k-instruct.Q4_K_M.gguf",
    n_ctx=1024,
    n_batch=64,
    n_threads=8,
    verbose=False
)

# model_path = "D:/Python projects/pdf_chatbot/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# llm = Llama(
#     model_path=model_path,
#     n_ctx=2048,     
#     n_threads=6,    
#     n_batch=128
# )

# llm = Llama(
#     model_path = "D:/Python projects/pdf_chatbot/models/llama-2-7b-chat.Q4_K_M.gguf",
#     n_ctx=8192,  
#     n_threads=8,
#     n_gpu_layers=0,
# )

DOC_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(DOC_EMBED_MODEL_NAME)
EMBED_DIM = embed_model.get_sentence_embedding_dimension()
print("Embedding dimension:", EMBED_DIM)

MAX_HISTORY = 50
REDIS_TTL_SECONDS = 60 

User = get_user_model()

INDEX_PATH = "D:/Python projects/pdf_chatbot/faiss_index"
os.makedirs(INDEX_PATH, exist_ok=True)

FAISS_DIR = os.path.join(settings.BASE_DIR, "faiss_indexes")
os.makedirs(FAISS_DIR, exist_ok=True)


redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

def is_session_valid(user_id, session_id):
    redis_key = f"user_session:{user_id}"
    active_session = redis_client.get(redis_key)
    return active_session == session_id



def extract_cv_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)

    return ""

def extract_candidate_name(cv_text):
    lines = [l.strip() for l in cv_text.splitlines() if l.strip()]

    for line in lines[:5]:  # top of CV
        if (
            len(line.split()) in range(2, 5) and
            line.isupper() and
            not any(k in line.lower() for k in ["email", "phone", "linkedin", "github"])
        ):
            return line.title()

    # fallback: email username
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', cv_text)
    if email_match:
        name_part = email_match.group(0).split("@")[0]          
        name_part = re.sub(r'\d+', '', name_part)             
        name = name_part.replace(".", " ").title()             
    return name

def search_google_serpapi(query, num_results=3):
    """Fetch top Google search results using SerpAPI."""
    api_key = settings.SERPAPI_KEY
    if not api_key:
        return []

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        results = []
        for item in data.get("organic_results", [])[:num_results]:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet", "")
            results.append({
                "title": title,
                "link": link,
                "snippet": snippet
            })
        return results

    except Exception as e:
        print(f"SerpAPI error: {e}")
        return []


def remove_emojis_and_special_chars(text):
    cleaned = re.sub(r'[^\x00-\x7F]+', '', text)
    return cleaned.strip()

def extract_text_from_file(file_path):
    text = ""
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            text = extract_text_from_pdf_with_fitz(file_path)
            
        elif ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8", engine="python")
            rows = df.astype(str).apply(lambda x: " | ".join(x), axis=1)
            text = "\n".join(rows)
            text = clean_extracted_text(text)

        elif ext in [".json", ".jsonl"]:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False, indent=2)
            text = clean_extracted_text(text)

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8-sig") as f:
                text = clean_extracted_text(f.read())

        else:
            text = "Unsupported file format."

    except Exception as e:
        text = f"Error reading file: {e}"

    return clean_extracted_text_preserve_lines(text.strip())


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


def clean_extracted_text(text):
    """Sanitize text while preserving paragraph structure."""
    if not text:
        return ""
  
    text = text.replace("\ufeff", "").replace("\x00", "")
  
    text = re.sub(r'(?i)copyright.*?all rights reserved.*', '', text)
    text = re.sub(r'(?i)page\s*\d+', '', text)
    text = re.sub(r'[•·●▪▶►\-\–\—_,]+', ' ', text)
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[“”"\'`´]', '', text)
    text = re.sub(r'[^\w\s.,;:!?()\n]', '', text) 
    text = re.sub(r'\b\d+(\.\d+)+\b', '', text)  
    text = re.sub(r"[^a-zA-Z0-9.,? \n]+", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
   
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


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


def build_faiss_index_from_text(text, index_path,uploaded_file=None, chunk_size=800, overlap=100):
    """
    Builds FAISS index + metadata with page and line references.
    Each chunk will include page_no, line_start, line_end info.
    """
    
    meta = []
    pages = text.split("--- PAGE BREAK ---")
    
    chunks = []
    page_no = 1
    line_counter = 0

    for page_text in pages:
        lines = page_text.strip().splitlines()
        cleaned_lines = [ln.strip() for ln in lines if ln.strip()]
        if not cleaned_lines:
            page_no += 1
            continue

        text_on_page = " ".join(cleaned_lines)
        sentences = re.split(r'(?<=[.!?]) +', text_on_page)

        current_chunk = ""
        start_line = 0

        for i, sentence in enumerate(sentences):
            if len(current_chunk) + len(sentence) < chunk_size:
                if not current_chunk:
                    start_line = line_counter
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                meta.append({
                    "page_no": page_no,
                    "line_start": start_line,
                    "line_end": line_counter,
                })
                current_chunk = sentence + " "
                start_line = line_counter
            line_counter += 1

        if current_chunk:
            chunks.append(current_chunk.strip())
            meta.append({
                "page_no": page_no,
                "line_start": start_line,
                "line_end": line_counter,
            })

        page_no += 1

   
    if not chunks:
        raise ValueError("No chunks generated for indexing.")

    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    faiss.write_index(index, index_path)
 
    meta_path = index_path.replace(".index", "_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({"chunks": chunks, "meta": meta}, f)

    print(f" Built FAISS index with {len(chunks)} chunks and metadata: {index_path}")
    if uploaded_file:
        FileChunk.objects.filter(file=uploaded_file).delete()

        total_chunks = len(chunks)
    try:    
        total_chunks =len(chunks)
        for i, chunk_text in enumerate(chunks):
            FileChunk.objects.create(
                file=uploaded_file,
                chunk_index=i,
                chunk_text=chunk_text,
                embedding=pickle.dumps(embeddings[i]),   
                index_path=index_path,
                total_chunks=total_chunks
            )
    except Exception as e:
        print("Error saving chunk:", e)

    return index_path


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


def deep_clean_answer(text: str) -> list:
    """
    Cleans and structures model-generated answers:
    - Removes noise (ISO codes, section/page numbers)
    - Fixes broken or spaced words
    - Normalizes punctuation and spacing
    - Returns properly formatted bullet sentences
    """
    if not text or not isinstance(text, str):
        return []

    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r'\b\d+(\.\d+)*\b', '', text)
    text = re.sub(r'\bpage\s*\d+\b', '', text, flags=re.I)
    text = re.sub(r'\[SOURCE:.*?\]', '', text, flags=re.I)
    text = re.sub(r'Provided by.*?ANSI', '', text, flags=re.I)
    text = re.sub(r'All rights reserved.*?ISO', '', text, flags=re.I)
    text = re.sub(r'No reproduction.*?license', '', text, flags=re.I)
    text = re.sub(r'(\.{2,}|,+|:+|;+|—+|–+)', '.', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:!?()\n ]', '', text)
    text = re.sub(r'\b\d{1,3}[A-Za-z]*\b', '', text)                  
    text = re.sub(r'\b\d+(\.\d+)*\b', '', text)                       
    text = re.sub(r'\([A-Z]\)|\b[A-Z]\)', '', text)
    text = re.sub(r'\bISO\b', '', text, flags=re.I) 
    text = re.sub(r"\(\s*\)", '', text)
    text = re.sub(r"[•●◦▪]", "", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b([A-Za-z])\s(?=[a-z])', r'\1', text)
    text = re.sub(r'(?<!\b(be|to|an|in|on|by|at|as|of|or|if|is|it|do|so|no))\b([A-Za-z])\s(?=[a-z])', r'\2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'(^|\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

    sentences = re.split(r'(?<=[.?!])\s+', text)
    bullets = [f"• {s.strip()}" for s in sentences if s.strip()]

    return bullets


def split_into_chunks(text, chunk_size=600, overlap=50):
    """
    Memory-safe splitter — processes text in small slices without loading all words in RAM.
    Suitable for 100+ page PDFs.
    """

    text = re.sub(r'[ \t]+', ' ', text.strip())

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks, current_chunk = [], []

    for sentence in sentences:
        current_chunk.append(sentence)
    
        joined = " ".join(current_chunk)
        if len(joined.split()) >= chunk_size:
            chunks.append(joined)
          
            overlap_words = joined.split()[-overlap:]
            current_chunk = overlap_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def home_view(request):
    return render(request, "home.html")


def signup_view(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, "Signup successful!")
            return redirect("login")
        else:
            messages.error(request, "Please fix the errors below.")
    else:
        form = SignUpForm()
    return render(request, "signup.html", {"form": form})


@never_cache
def login_view(request):
    if request.method == "POST":
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            request.session['chat_history'] = []
            messages.success(request, "Login successful!")

            if user.is_superuser: 
                return redirect('admin')

            if getattr(user, "is_main", False):
                return redirect('main')
            else:
                return redirect('chatbot')

        else:
            messages.error(request, "Invalid credentials.")
            return redirect('login')

    else:
        form = LoginForm()

    resp = render(request, "login.html", {"form": form})
    resp['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp['Pragma'] = 'no-cache'
    resp['Expires'] = '0'
    return resp



@never_cache
def logout_view(request):
    if 'chat_history' in request.session:
        del request.session['chat_history']
    logout(request)
    request.session.flush()
    response = redirect('login')
    response.delete_cookie('sessionid')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response


def check_session(request):
    return JsonResponse({'logged_in': request.user.is_authenticated})


@login_required(login_url='login')
@never_cache
def main_view(request):
    if not request.user.is_authenticated:
        return redirect('login')

    if not getattr(request.user, 'is_main', False):
        messages.error(request, "Access denied.")
        return redirect("login")

    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        category = request.POST.get("category", "General")

        if uploaded_file:
          
            existing_file = UploadedFile.objects.filter(
                original_name=uploaded_file.name,
                uploaded_by=request.user
            ).first()
            if existing_file:
                
                delete_faiss_index(existing_file.id, request.user.id)
                existing_file.delete()

            
            uf = UploadedFile.objects.create(
                uploaded_by=request.user,
                file=uploaded_file,
                original_name=uploaded_file.name,
                category=category
            )

            extracted_text = extract_text_from_pdf_with_fitz(uf.file.path)
            extracted_text = "\n".join(sent_tokenize(extracted_text))
            uf.extracted_text = extracted_text.strip() 
            uf.save()

            index_name = f"user_{request.user.id}_file_{uf.id}.index"
            index_path = os.path.join(FAISS_DIR, index_name)
            build_faiss_index_from_text(extracted_text, index_path,uploaded_file=uf)

            print(f" File '{uf.original_name}' uploaded and indexed for user {request.user.username}")
            messages.success(request, f"File '{uf.original_name}' uploaded and indexed successfully!")
            return redirect("main")

    
    uploaded_files = UploadedFile.objects.filter(uploaded_by=request.user).order_by('-uploaded_at')
    return render(request, "main.html", {"uploaded_files": uploaded_files})

@login_required(login_url='login')
@never_cache
def chatbot_view(request):
    """Chatbot powered by FAISS + LLaMA (local) — conceptual + factual accuracy optimized."""
    User = get_user_model()
    response_text = ""
    debug_info = ""
    chat_history = []
    reference_items = []
    best_file = None

    REDIS_TTL_SECONDS = 86400
    redis_key = f"chat_history:{request.user.id}"

    try:
        raw = cache.get(redis_key)
        chat_history = json.loads(raw) if raw else []
    except Exception:
        chat_history = []

    # main_user = User.objects.filter(is_main=True).first()

    if request.method == "POST":
        q = request.POST.get("question", "").strip()

        if not q:
            response_text = "Please enter a question."
        else:
            try:
                all_files = UploadedFile.objects.filter(extracted_text__isnull=False)
               
                best_chunks = None
                best_score = -1

                for f in all_files:
                    file_owner_id = f.uploaded_by.id
                    index_name = f"user_{file_owner_id}_file_{f.id}.index"
                    index_path = os.path.join(FAISS_DIR, index_name)
                    meta_path = index_path.replace(".index", "_meta.pkl")

                    if not os.path.exists(index_path) or not os.path.exists(meta_path):
                        build_faiss_index_from_text(f.extracted_text, index_path)

                    index, chunks, meta = load_faiss_index(index_path)
                    q_emb = embed_model.encode([q], convert_to_numpy=True).astype("float32")
                    D, I = index.search(q_emb, 5)

                    distances = []
                    for d, idx in zip(D[0], I[0]):
                        if idx >= 0 and idx < len(chunks):
                            distances.append(float(d))

                    if not distances:
                        continue

                    d_min, d_max = float(np.min(distances)), float(np.max(distances))
                    similarities = [1 - ((d - d_min) / (d_max - d_min + 1e-9)) for d in distances]
                    avg_sim = float(np.mean(similarities))

                    if avg_sim > best_score:
                        best_score = avg_sim
                        best_file = f
                        best_chunks = [chunks[idx] for idx in I[0] if idx >= 0 and idx < len(chunks)]

                if not best_file or not best_chunks or best_score < 0.15:
                    response_text = "Information not clearly found in the documents."
                else:
                    context = "\n\n".join(best_chunks[:3]).strip()

                    conceptual_keywords = ["benefit", "purpose", "role", "importance", "use", "impact", "objective"]
                    if any(word in q.lower() for word in conceptual_keywords):
                        q_prompt = "[Conceptual] " + q
                    else:
                        q_prompt = q

                    prompt = f"""
You are an expert. Answer accurately using ONLY the document context.

**Context:**
{context}

**Question:** {q_prompt}

**Guidelines:**
✓ Answer based only on provided context
✓ Use bullet points for clarity
✓ Explain requirements clearly
✓ If not in context: "Not specified in the provided document"

**Answer:**
"""

                    try:
                        if len(prompt) > 3500:
                            prompt = prompt[-3500:]
                        # res = llm(prompt=prompt, max_tokens=400, temperature=0.2)
                    except Exception as e:
                        try:
                            short_prompt = prompt[-1500:]
                            # res = llm(prompt=short_prompt, max_tokens=300, temperature=0.3)
                        except Exception as e2:
                            res = {"content": "Model failed internally or context too large."}

                    if isinstance(res, dict):
                        if "choices" in res and len(res["choices"]) > 0:
                            answer = res["choices"][0].get("text", "").strip()
                        elif "content" in res:
                            answer = res["content"].strip()
                        else:
                            answer = str(res)
                    else:
                        answer = str(res).strip()

                    answer = re.sub(r"^.*?--- Short Answer ---", "", answer, flags=re.DOTALL).strip()
                    lines = [line.lstrip("- ").strip() for line in answer.split("\n") if line.strip()]
                    lines = ["• " + line for line in lines]
                    answer = "\n".join(lines)

                    if not answer or len(answer.split()) < 5:
                        answer = "Information not clearly found in the document."

                    answer += f"\n\nSource: {best_file.original_name}"

                    response_text = answer

                    ChatbotQA.objects.create(
                        user=request.user,
                        uploaded_file=best_file,
                        question=q,
                        answer=response_text.strip()
                    )


                chat_history.append({
                    "question": q,
                    "answer": response_text.strip(),
                    "ts": timezone.now().isoformat()
                })
                cache.set(redis_key, json.dumps(chat_history), timeout=REDIS_TTL_SECONDS)

            except Exception as exc:
                response_text = f"Error: {exc}\n{traceback.format_exc()}"

    return render(request, "chatbot.html", {
        "response": response_text,
        "debug": debug_info,
        "chat_history": chat_history,
        "uploaded_file": best_file,
        "reference_items": reference_items,
    })




@login_required(login_url='login')
@never_cache
def admin_dashboard(request):
    if not request.user.is_superuser:
        messages.error(request, "Access denied.")
        return redirect('login')

    users = User.objects.all()
    files = UploadedFile.objects.all()
    chats = ChatbotQA.objects.all()
    chunks = FileChunk.objects.all().order_by('file','chunk_index')

    return render(request, "admin.html", {
        "users": users,
        "files": files,
        "chats": chats,
        "chunks": chunks
    })


@login_required(login_url='login')
@never_cache
def admin_create_user(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        is_main = request.POST.get("is_main") =="on"

        default_password = "123456789"

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
        else:
            user = User.objects.create_user(username=username, email=email, password=default_password)
            user.is_main = is_main
            user.save()
            messages.success(request, f"User {username} created successfully!")
    return redirect('admin')

@login_required(login_url='login')
@never_cache
def admin_delete_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    user.delete()
    messages.success(request, f"User deleted successfully!")
    return redirect('admin')


@login_required(login_url='login')
@never_cache
def admin_update_user(request, user_id):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        is_main = request.POST.get('is_main') == 'on'

        try:
            user = User.objects.get(id=user_id)
            user.username = username
            user.email = email
            user.is_main = is_main
            user.save()
            return redirect('admin')
        except User.DoesNotExist:
            messages.error(request, "User not found.")
            return redirect('admin')
    return redirect('admin')


@login_required(login_url='login')
@never_cache
def admin_create_file(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get('file')
        category = request.POST.get('category')
        username = request.POST.get('username')  

        if uploaded_file and category and username:
            try:
                user = User.objects.get(username=username, is_main=True) 
                UploadedFile.objects.create(
                    uploaded_by=user,  
                    file=uploaded_file,
                    original_name=uploaded_file.name,
                    category=category
                )
                messages.success(request, f"File uploaded successfully for {user.username}!")
            except User.DoesNotExist:
                messages.error(request, "Selected user does not exist or is not a main user.")
            return redirect('admin')

        messages.error(request, "Please select a file, category, and user.")
        return redirect('admin')

    return redirect('admin')


@login_required(login_url='login')
@never_cache
def admin_update_file(request):
    if request.method == 'POST':
        file_id = request.POST.get('file_id')
        category = request.POST.get('category')
        file_obj = request.FILES.get('file')  

        try:
            f = UploadedFile.objects.get(id=file_id)
            f.category = category

            if file_obj:
                if f.file:
                    f.file.delete(save=False)  
                f.file = file_obj
                f.original_name = file_obj.name 

            f.save()
            messages.success(request, "File updated successfully!")

        except UploadedFile.DoesNotExist:
            messages.error(request, "File not found!")

        return redirect('admin')  

    return redirect('admin')


@login_required(login_url='login')
@never_cache
def admin_delete_file(request, file_id):
    file_obj = get_object_or_404(UploadedFile, id=file_id)
    file_obj.delete()
    messages.success(request, "File deleted successfully!")
    return redirect('admin')

@login_required(login_url='login')
def admin_create_chat(request):
    if request.method == "POST":
        user_id = request.POST.get('user_id')
        file_id = request.POST.get('file_id')
        question = request.POST.get('question')
        answer = request.POST.get('answer')
        session_id = request.POST.get('session_id') or "default"

        if not user_id or not question or not answer:
            messages.error(request, "Please fill all required fields")
            return redirect('admin')

        user = get_object_or_404(User, id=user_id)
        uploaded_file = UploadedFile.objects.filter(id=file_id).first() if file_id else None

        ChatbotQA.objects.create(
            user=user,
            username=user.username,
            session_id=session_id,
            uploaded_file=uploaded_file,
            question=question,
            answer=answer
        )
        messages.success(request, "Chat created successfully!")
        return redirect('admin')


@login_required(login_url='login')
def admin_update_chat(request, chat_id):
    chat = get_object_or_404(ChatbotQA, id=chat_id)
    if request.method == "POST":
        user_id = request.POST.get('user_id')
        file_id = request.POST.get('file_id')
        question = request.POST.get('question')
        answer = request.POST.get('answer')

        user = get_object_or_404(User, id=user_id)
        uploaded_file = UploadedFile.objects.filter(id=file_id).first() if file_id else None

        chat.user = user
        chat.username = user.username
        chat.uploaded_file = uploaded_file
        chat.question = question
        chat.answer = answer
        chat.save()

        messages.success(request, "Chat updated successfully!")
        return redirect('admin')


@login_required(login_url='login')
def admin_delete_chat(request, chat_id):
    chat = get_object_or_404(ChatbotQA, id=chat_id)
    chat.delete()
    messages.success(request, "Chat deleted successfully!")
    return redirect('admin')


@login_required
def admin_delete_chunk(request, chunk_id):
    chunk = get_object_or_404(FileChunk, id=chunk_id)
    chunk.delete()
    messages.success(request, "Chunk deleted successfully.")
    return redirect('admin')

@login_required
def admin_edit_chunk(request, chunk_id):
    chunk = get_object_or_404(FileChunk, id=chunk_id)
    if request.method == "POST":
        chunk_text = request.POST.get("chunk_text")
        chunk.chunk_text = chunk_text
        chunk.save()
        messages.success(request, "Chunk updated successfully.")
        return redirect('admin')


@csrf_exempt
def api_login(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    username = data.get("username")
    email = data.get("email")

    if not username or not email:
        return JsonResponse({"error": "username and email are required"}, status=400)
    username_exists = ApiUser.objects.filter(username=username).exists()
    email_exists = ApiUser.objects.filter(email=email).exists()

    if username_exists and not email_exists:
        return JsonResponse({"error": "Username already exists, please use a different one."}, status=400)
    elif email_exists and not username_exists:
        return JsonResponse({"error": "Email already exists, please use a different one."}, status=400)
    elif username_exists and email_exists:
      
        user = ApiUser.objects.filter(username=username, email=email).first()

   
    user, created = ApiUser.objects.get_or_create(
        username=username,
        email=email,
        defaults={"is_main": False}  
    )

 
    main_session_key = f"user_session:{user.id}"
    page_access_key = f"user_page_access:{user.id}"

   
    existing_session = redis_client.get(main_session_key)
    if existing_session:
        session_id = existing_session.decode() if isinstance(existing_session, bytes) else existing_session
        main_ttl = redis_client.ttl(main_session_key)
        page_access = redis_client.get(page_access_key)
        page_ttl = redis_client.ttl(page_access_key) if page_access else 0

      
        if user.is_main or page_access:
            redirect_url = "/api/upload/"
        else:
            redirect_url = "/api/chat/"

        return JsonResponse({
            "message": "You already have an active session. Please use the existing session_id.",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "session_id": session_id,
                "is_main": user.is_main,
                "session_expiry_seconds": main_ttl,
                "page_access_expiry_seconds": page_ttl if page_access else 0
            },
            "redirect_to": redirect_url
        }, status=200)

  
    new_session = uuid.uuid4().hex
    user.session_id = new_session
    user.save(update_fields=["session_id"])

    redis_client.setex(main_session_key, 86400, new_session)

   
    if user.is_main:
        redis_client.setex(page_access_key, 3600, "upload_access")

    redirect_url = "/api/upload/" if user.is_main else "/api/chat/"

    return JsonResponse({
        "message": "User Created Succesfully.",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "session_id": new_session,
            "is_main": user.is_main,
            "session_expiry_seconds": 86400,
            "page_access_expiry_seconds": 3600 if user.is_main else 0,
            "new_user_created": created
        },
        "redirect_to": redirect_url
    }, status=200)



@csrf_exempt
def api_upload_file(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    id = data.get("id")
    session_id = data.get("session_id")
    file_url = data.get("file_url")  
    file_name = data.get("file_name")
    category = data.get("category", "General")

    if not id or not session_id or not file_url or not file_name:
        return JsonResponse({
            "error": "id, session_id, file_url, and file_name are required"
        }, status=400)
    
    redis_key = f"user_session:{id}"
    stored_session = redis_client.get(redis_key)
    if not stored_session or stored_session != session_id:
        return JsonResponse({"error": "Your session has expired. Please log in again."}, status=401)

    try:
        user = ApiUser.objects.get(id=id, session_id=session_id)
    except ApiUser.DoesNotExist:
        return JsonResponse({"error": "Invalid user or session ID"}, status=403)

    if not user.is_main:
        return JsonResponse({"error": "Permission denied. Only main users can upload files."}, status=403)

 
    existing_file = ApiUploadedFile.objects.filter(
        uploaded_by=user,
        original_name=file_name
    ).first()

    if existing_file:
        # Optional: delete FAISS index if used
        # delete_faiss_index(existing_file.id, user.id)

        # Delete previous file record
        existing_file.delete()

    
    uf = ApiUploadedFile.objects.create(
        uploaded_by=user,
        original_name=file_name,
        category=category,
    )

   
    if file_url.startswith("http"):
        import requests
        response = requests.get(file_url)
        if response.status_code != 200:
            return JsonResponse({"error": "Failed to download file from URL"}, status=400)
        uf.file.save(file_name, ContentFile(response.content))
    else:
        try:
            with open(file_url, "rb") as f:
                uf.file.save(file_name, ContentFile(f.read()))
        except FileNotFoundError:
            return JsonResponse({"error": f"Local file not found: {file_url}"}, status=400)

  
    extracted_text = extract_text_from_pdf_with_fitz(uf.file.path)
    uf.extracted_text = extracted_text
    uf.save()

    return JsonResponse({
        "status": "success",
        "message": "File uploaded successfully",
        "file": {
            "id": uf.id,
            "name": uf.original_name,
            "category": uf.category
        }
    })


def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",  
        "prompt": prompt,
        "stream": False
    }
    try:
        
        response = requests.post(url, json=payload, timeout=200)
        response.raise_for_status()
        result = response.json()

        return result.get("response", "").strip()

    except requests.exceptions.ReadTimeout:
        return "Model took too long to respond (timeout). Try again."
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Make sure 'ollama serve' is running."
    except Exception as e:
        return f"Ollama error: {e}"


@csrf_exempt
def api_chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    user_id = data.get("user_id")
    session_id = data.get("session_id")
    question = data.get("question", "").strip()

    if not user_id or not session_id:
        return JsonResponse({"error": "user_id and session_id required"}, status=400)
    if not question:
        return JsonResponse({"error": "question required"}, status=400)


    redis_key = f"user_session:{user_id}"
    stored_session = redis_client.get(redis_key)
    if not stored_session:
        return JsonResponse({"error": "Your session has expired. Please log in again."}, status=401)

    if isinstance(stored_session, bytes):
        stored_session = stored_session.decode()

    if stored_session != session_id:
        return JsonResponse({"error": "Invalid or expired session. Please login again."}, status=401)


    page_access_key = f"user_page_access:{user_id}"
    has_upload_access = bool(redis_client.get(page_access_key))

    try:
        api_user = ApiUser.objects.get(id=user_id, session_id=session_id)
    except ApiUser.DoesNotExist:
        return JsonResponse({"error": "Invalid user_id or session ID"}, status=403)

    if api_user.is_main or has_upload_access:
        main_user = api_user
        uploaded_file = ApiUploadedFile.objects.filter(uploaded_by=main_user).order_by("-uploaded_at").first()
    else:
        main_user = ApiUser.objects.filter(is_main=True).first()
        if not main_user:
            return JsonResponse({"error": "No main user found to fetch file."}, status=404)
        uploaded_file = ApiUploadedFile.objects.filter(uploaded_by=main_user).order_by("-uploaded_at").first()

    if not uploaded_file or not uploaded_file.extracted_text:
        return JsonResponse({"error": "No uploaded file found. Please ask main user to upload one."}, status=404)

    try:

        index_name = f"user_{main_user.id}_file_{uploaded_file.id}.index"
        index_path = os.path.join(FAISS_DIR, index_name)
        meta_path = index_path.replace(".index", "_meta.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            build_faiss_index_from_text(uploaded_file.extracted_text, index_path)

        index, chunks, meta = load_faiss_index(index_path)
        q_emb = embed_model.encode([question], convert_to_numpy=True).astype("float32")
        D, I = index.search(q_emb, 5)

        distances, retrieved, retrieved_meta = [], [], []
        for idx, d in zip(I[0], D[0]):
            if 0 <= idx < len(chunks):
                retrieved.append(chunks[idx])
                retrieved_meta.append(meta[idx])
                distances.append(float(d))

        if not retrieved:
            answer = "Information not clearly found in the document."
        else:
            d_min, d_max = float(np.min(distances)), float(np.max(distances))
            similarities = [1 - ((d - d_min) / (d_max - d_min + 1e-9)) for d in distances]
            avg_sim = float(np.mean(similarities))

            if avg_sim < 0.15:
                answer = "Information not clearly found in the document."
            else:
                context = "\n\n".join(retrieved[:3]).strip()
                prompt = f"""
You are an ISO 22301:2019 BCMS specialist. Answer accurately using ONLY the document context.

**Standard:** ISO 22301:2019 - Business Continuity Management Systems

**Context:**
{context}

**Question:** {question}

**Guidelines:**
✓ Answer based only on provided context  
✓ Use bullet points for clarity  
✓ Include clause numbers (e.g., Clause 8.2)  
✓ Explain requirements clearly  
✓ If not in context: "Not specified in the provided ISO 22301:2019 sections"

**Answer:**
"""

             
                answer = query_ollama(prompt)

                answer = re.sub(r"\s+", " ", answer).strip()
                if not answer or len(answer.split()) < 5:
                    answer = "Information not clearly found in the document."

        google_results = search_google_serpapi(question)                    

   
        ApiChatMessage.objects.create(
            user=api_user,
            session_id=session_id,
            question=question,
            answer=answer
        )

        return JsonResponse({
            "status": "success",
            "question": question,
            "answer": answer,
            "references": google_results
        }, status=200)

    except Exception as exc:
        return JsonResponse({
            "status": "error",
            "message": str(exc),
            "trace": traceback.format_exc()
        }, status=500)


@csrf_exempt
def api_chat_history(request):
    if request.method != "GET":
        return JsonResponse({"error": "Only GET allowed"}, status=405)

    user_id = request.GET.get("user_id")
    session_id = request.GET.get("session_id")

    if not user_id or not session_id:
        return JsonResponse({"error": "user_id and session_id required"}, status=400)

    try:
        api_user = ApiUser.objects.get(id=user_id, session_id=session_id)
    except ApiUser.DoesNotExist:
        return JsonResponse({"error": "Invalid user_id or session_id"}, status=403)

    chats = ApiChatMessage.objects.filter(
        user=api_user,
        session_id=session_id
    ).order_by("created_at")

   
    chat_history = [
        {
            "question": chat.question,
            "answer": chat.answer,
            "references": chat.references or [],
            "created_at": chat.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for chat in chats
    ]

    return JsonResponse({
        "status": "success",
        "user_id": api_user.id,
        "session_id": session_id,
        "total_chats": len(chat_history),
        "chat_history": chat_history
    }, status=200)


@login_required
def interview_home(request):
    return render(request, "interview_home.html", {
        "roles": [
            "Python Developer",
            "Frontend Developer",
            "AI/ML Engineer",
            "Cybersecurity Analyst",
            "Data Analyst",
            "Custom"
        ]
    })

@login_required
def interview_start(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    cv_file = request.FILES.get("cv_file")
    if not cv_file:
        return HttpResponseBadRequest("CV file required")

    cv_text = extract_cv_text(cv_file)
    candidate_name = extract_candidate_name(cv_text)
    difficulty = request.POST.get("difficulty", "medium")
    q_count = int(request.POST.get("question_count", 8))

    session = InterviewSession.objects.create(
        user=request.user,
        role="CV-Based Interview",
        metadata={
            "difficulty": difficulty,
            "question_count": q_count,
            "candidate_name": candidate_name,
        },
        cv_text=cv_text
    )

    session.mark_started()

    cv_short = cv_text[:2000]

    prompt = f"""
You are an expert AI that ONLY outputs JSON arrays of interview questions.
STRICT RULES:
1. Output MUST be valid JSON.
2. Output MUST start with '[' and end with ']'.
3. NO extra text, explanations, or formatting outside JSON.
4. Each question object MUST have:
   - "q": the question text (string)
   - "expected": short expected answer (string)
5. Generate EXACTLY {q_count} questions based on the CV below.

CV:
{cv_short}

OUTPUT EXAMPLE:
[
  {{"q": "Example question?", "expected": "Short expected answer."}}
]
"""

    questions_data = []

    try:
        llm_result = LLM_MODEL(prompt, max_tokens=500, temperature=0.2)
        raw = llm_result["choices"][0]["text"].strip()
        print("LLM RAW OUTPUT:", raw)

        raw = re.sub(r"^-?\s*\[?response\]?:", "", raw, flags=re.I).strip()
        start = raw.find("[")
        if start == -1:
            raise ValueError("JSON array not found")

        raw = raw[start:]

        objects = re.findall(r"\{[^{}]*\}", raw, re.S)

        for obj in objects:
            try:
                parsed = json.loads(obj)
                if "q" in parsed:
                    questions_data.append(parsed)
            except:
                continue

        questions_data = questions_data[:q_count]

    except Exception as e:
        print("LLM Error:", e)
        questions_data = []

    if not questions_data:
        questions_data = [
            {"q": f"Default Question {i+1} based on CV", "expected": ""}
            for i in range(q_count)
        ]

    for idx, q in enumerate(questions_data):
        InterviewQuestion.objects.create(
            session=session,
            order=idx + 1,
            text=q.get("q", f"Question {idx+1}"),
            expected=q.get("expected", "")
        )

    return redirect("interview_process")



@login_required
def interview_process(request):
    session = InterviewSession.objects.filter(user=request.user, status="in_progress").order_by("-created_at").first()
    if not session:
        return redirect("interview_home")

    next_q = session.questions.filter(responses__isnull=True).first()

    if not next_q:
        return redirect("interview_finish")

    return render(request, "interview_process.html", {
        "session": session,
        "question": next_q
    })


@login_required
def interview_next_question(request):
    session = InterviewSession.objects.filter(user=request.user, status="in_progress").order_by("-created_at").first()
    if not session:
        return render(request, "interview_question.html", {"question": None})

    next_q = session.questions.filter(responses__isnull=True).first()
    return render(request, "interview_question.html", {"question": next_q})

MAX_TOKENS = 1500  

@login_required
def interview_submit_answer(request):
    if request.method != "POST":
        return redirect("interview_process")

    session = InterviewSession.objects.filter(user=request.user, status="in_progress").order_by("-created_at").first()
    if not session:
        return redirect("interview_home")
    print("POST DATA:", request.POST) 
    question_id = request.POST.get("question_id")
    answer_text = request.POST.get("final_answer")
    print("RAW POST:", request.POST)
    if answer_text:
        answer_text = answer_text.strip()
    else:
        answer_text = ""

    question = get_object_or_404(InterviewQuestion, id=question_id, session=session)

    score = 0.0
    feedback = "No answer provided."

    if answer_text:
        cv_text = session.cv_text or ""
        cv_short = cv_text[:1500]
        q_text = question.text[:500]
        ans_text = answer_text[:1000]

        llm_prompt = f"""
Score the answer from 0 to 10.
Return ONLY JSON in this format:
{{"score": number, "feedback": "short feedback"}}

Question: {q_text}
Answer: {ans_text}
"""

        try:
            llm_result = LLM_MODEL(llm_prompt, max_tokens=80, temperature=0.0)

            raw_text = llm_result["choices"][0]["text"].strip()
            match = re.search(r'\{.*?\}', raw_text, re.DOTALL)

            if match:
                eval_data = json.loads(match.group(0))
                score = float(eval_data.get("score", 5.0))
                feedback = eval_data.get("feedback", "No feedback provided.")
            else:
                print("No valid JSON found in LLM output, using placeholder.")
                score = 5.0
                feedback = "Auto-evaluation placeholder."

        except Exception as e:
            print("LLM scoring error:", e)
            score = 5.0
            feedback = "Auto-evaluation placeholder."


    response_obj, created = InterviewResponse.objects.update_or_create(
    question=question,
    defaults={
        "answer_text": answer_text,
        "score": score,
        "feedback": feedback
    }
)

    print(f"Saved response: QID={question.id}, Answer={answer_text}, Score={score}, Feedback={feedback}")

    return redirect("interview_process")



@login_required
def interview_finish(request):
    session = InterviewSession.objects.filter(user=request.user, status="in_progress").order_by("-created_at").first()
    if not session:
        return redirect("interview_home")

    total, count = 0, 0
    for q in session.questions.all():
        r = q.responses.first()
        if r:
            total += r.score
            count += 1

    session.total_score = total / count if count else 0
    session.mark_completed()
    session.metadata["answered_count"] = count
    session.save()

    return redirect("interview_report")

@login_required
def interview_report(request):
    session = InterviewSession.objects.filter(user=request.user).order_by("-created_at").first()
    if not session:
        return redirect("interview_home")

    questions = session.questions.all().prefetch_related("responses")
    return render(request, "interview_report.html", {"session": session, "questions": questions})