from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.core.cache import cache
from django.http import JsonResponse
from django.contrib import messages
from datetime import datetime
from .forms import SignUpForm, LoginForm
from .models import UploadedFile, ChatbotQA, CustomUser
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import timedelta
import os, json, csv, fitz, nltk
from .utils import (
    extract_file_content,
    make_sentence_embeddings,
    find_best_sentence_answer
)
import faiss
import traceback
import pickle
from django.conf import settings
import numpy as np
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import re
from llama_cpp import Llama 
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize,PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters

nltk.data.path.append(r'D:\Python projects\pdf_chatbot-1\pdf_chat\nltk_data')
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, download_dir=r"D:\Python projects\pdf_chatbot-1\pdf_chat\nltk_data")
nltk.download('stopwords')

punkt_param = PunktParameters()
tokenizer = PunktSentenceTokenizer(punkt_param)


# embed_model = SentenceTransformer("Adel-Elwan/msmarco-bert-base-dot-v5-fine-tuned-AI")

llm = Llama(
    model_path = "D:/Python projects/pdf_chatbot-1/models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,  
    n_threads=8,
    n_gpu_layers=0,
)

DOC_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(DOC_EMBED_MODEL_NAME)
EMBED_DIM = embed_model.get_sentence_embedding_dimension()
print("Embedding dimension:", EMBED_DIM)

MAX_HISTORY = 50
REDIS_TTL_SECONDS = 60 



INDEX_PATH = "D:/Python projects/pdf_chatbot-1/faiss_index"
os.makedirs(INDEX_PATH, exist_ok=True)

LOCAL_LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"  



FAISS_DIR = os.path.join(settings.BASE_DIR, "faiss_indexes")
os.makedirs(FAISS_DIR, exist_ok=True)


def remove_emojis_and_special_chars(text):
    cleaned = re.sub(r'[^\x00-\x7F]+', '', text)
    return cleaned.strip()


def extract_text_from_file(file_path):
    import pandas as pd
    text = ""
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            text = extract_text_from_pdf_with_fitz(file_path)
            text = re.sub(r"(?i)copyright.*?reserved", "", text)
            text = re.sub(r"(?i)licensee=.*", "", text)
            text = re.sub(r"(?i)no reproduction.*", "", text)
            text = re.sub(r"ISO\s*\d{3,5}(:\d{4})?", "", text)
            text = re.sub(r"[`\-,‘’“”\'\"•▪●★→⇒✓■□△▲▶▶️❖\*]+", " ", text)
            text = re.sub(r"\bPage\s*\d+\b", "", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
            
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


def clean_extracted_text_preserve_lines(text):
    """Clean text lightly but keep paragraph and line structure."""
    text = text.replace("\ufeff", "").replace("\x00", "")
    text = re.sub(r'(?i)copyright.*?(iso|organization|standardization).*', '', text)
    text = re.sub(r'(?i)no reproduction.*', '', text)
    text = re.sub(r'(?i)licensee.*', '', text)
    text = re.sub(r'(?i)provided by.*', '', text)
    text = re.sub(r'(?i)all rights reserved.*', '', text)
    text = re.sub(r'ISO\s*\d{3,5}(:\d{4})?', '', text)
    text = re.sub(r'(?i)page\s*\d+', '', text)
    text = re.sub(r'(?i)reference number.*', '', text)
    text = re.sub(r'(?i)(contents|table of contents|bibliography|foreword|introduction)\s+.*', '', text)
    text = re.sub(r'Licensee=.*', '', text)
    text = re.sub(r'Not for Resale.*', '', text)
    text = re.sub(r'---+', '', text)
    text = re.sub(r'^\s*\d+(\.\d+)*\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[•·●▪▶►\-\–\—_,]+', ' ', text)
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'[“”"\'`´]', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,? \n]+', ' ', text)
    text = re.sub(r'\b[vx]{1,4}\b', '', text)
    text = re.sub(r'\(\s*\)', '', text)  
    text = re.sub(r'\b(E|F|G|A|B)\b(?=\))', '', text)  
    text = re.sub(r'::+', ':', text)
    text = re.sub(r' +', ' ', text)  
    text = re.sub(r'(?<=\b[A-Z]) (?=[A-Z]+\b)', '', text) 
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\bNOTE\b.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\b[a-zA-Z])\s(?=[a-zA-Z]{2,}\b)", "", text)
    text = re.sub(r'\b([A-Z])\s+([a-z]{2,})\b', r'\1\2', text)
    


    return text.strip()


def build_faiss_index_from_text(text, index_path, chunk_size=600, overlap=50):
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

def deep_clean_answer(text: str) -> str:
    """
    Cleans text from unwanted patterns:
    - ISO codes, section numbers, page numbers
    - Source references like [SOURCE: ...]
    - Excess punctuation, special symbols
    - Converts text to clean bullet points format
    """
    if not text or not isinstance(text, str):
        return []
    text = text.replace("\r", " ")
    text = re.sub(r'\b\d+(\.\d+)*\b', '', text)
    text = re.sub(r'\bpage\s*\d+\b', '', text, flags=re.I)
    text = re.sub(r'\[SOURCE:.*?\]', '', text, flags=re.I)
    text = re.sub(r'(\.{2,})', '.', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:!?()\n ]', '', text)
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'(\.{2,}|,+|:+|;+|—+|–+)', '.', text)  
    text = re.sub(r'\(\s*\)', '', text)  
    text = re.sub(r'\b(E|F|G|A|B)\b(?=\))', '', text)  
    text = re.sub(r'::+', ':', text)  
    text = re.sub(r' +', ' ', text) 
    text = re.sub(r'(?<=\b[A-Z]) (?=[A-Z]+\b)', '', text)  
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'Provided by.*?ANSI', '', text, flags=re.I)
    text = re.sub(r"\b\d{1,3}\s*[.)]?\s*", "", text)
    text = re.sub(r"(?m)^\s*[\-\*\•\·\●\▪\▶]+\s*", "", text)
    text = re.sub(r'All rights reserved.*?ISO', '', text, flags=re.I)
    text = re.sub(r'No reproduction.*?license', '', text, flags=re.I)
    text = re.sub(r'[^a-zA-Z0-9.,;:!?()\n ]', '', text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[•●◦▪]", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s*([a-d]\))", r"\n\1", text)
    text = re.sub(r"(?m)^\s*[\-\*\•\·\●\▪\▶]+\s*", "", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"[\-–]{2,}", "–", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=\b[a-zA-Z])\s(?=[a-zA-Z]{2,}\b)", "", text)
    text = re.sub(r"(^|\.\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    
    text = text.strip()

    sentences = tokenizer.tokenize(text)
    bullets = [f"• {s.strip()}" for s in sentences if s.strip()]

    return bullets




def split_into_chunks(text, chunk_size=600, overlap=50):
    """
    Memory-safe splitter — processes text in small slices without loading all words in RAM.
    Suitable for 100+ page PDFs.
    """
    import re

 
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
            if getattr(user, "is_main", False):
                return redirect('main')
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

            extracted_text = deep_clean_answer(uf.file.path)
            extracted_text = "\n".join(sent_tokenize(extracted_text))
            uf.extracted_text = extracted_text
            uf.save()

            index_name = f"user_{request.user.id}_file_{uf.id}.index"
            index_path = os.path.join(FAISS_DIR, index_name)
            build_faiss_index_from_text(extracted_text, index_path)

            print(f" File '{uf.original_name}' uploaded and indexed for user {request.user.username}")
            messages.success(request, f"File '{uf.original_name}' uploaded and indexed successfully!")
            return redirect("main")

    
    uploaded_files = UploadedFile.objects.filter(uploaded_by=request.user).order_by('-uploaded_at')
    return render(request, "main.html", {"uploaded_files": uploaded_files})



@login_required(login_url='login')
@never_cache
def chatbot_view(request):
    """Render chatbot page and answer questions using FAISS + LLaMA (local)."""
    User = get_user_model() 
    last_chat = None
    response_text = ""
    debug_info = ""
    reference_items = []
    try:
        expiry_time = timezone.now() - timedelta(minutes=1)
        db_chat_history = list(ChatbotQA.objects.filter(user=request.user).order_by("-created_at")[:MAX_HISTORY])
    except Exception:
        db_chat_history = []

    main_user = User.objects.filter(is_main=True).first()

    if request.user.is_main:
        
        uploaded_file = UploadedFile.objects.filter(uploaded_by=request.user).order_by("-uploaded_at").first()
    else:
        
        uploaded_file = UploadedFile.objects.filter(uploaded_by=main_user).order_by("-uploaded_at").first()


       
    redis_key = f"chat_history:{request.user.id}"
    try:
        raw = cache.get(redis_key)
        if raw:
           
            chat_history = json.loads(raw)
        else:
           
            chat_history = [
                {"question": c.question, "answer": c.answer, "ts": c.created_at.isoformat() if getattr(c, "created_at", None) else None}
                for c in db_chat_history
            ]
    except Exception:
        
        chat_history = [
            {"question": c.question, "answer": c.answer, "ts": c.created_at.isoformat() if getattr(c, "created_at", None) else None}
            for c in db_chat_history
        ]

    last_chat = chat_history[0] if chat_history else None

    if request.method == "POST":
        q = request.POST.get("question", "").strip()
        if not q:
            response_text = "Please enter a question."
        elif not uploaded_file or not uploaded_file.extracted_text:
            response_text = "No uploaded file found. Please upload a file first."
        else:
            try:
                index_name = f"main_user_{main_user.id}_file_{uploaded_file.id}.index"
                index_path = os.path.join(FAISS_DIR, index_name)
                meta_path = index_path.replace(".index", "_meta.pkl")

                if not os.path.exists(index_path) or not os.path.exists(meta_path):
                 
                    cleaned_text = uploaded_file.extracted_text or ""
                    build_faiss_index_from_text(cleaned_text, index_path)

                index, chunks, meta = load_faiss_index(index_path)
                q_emb = embed_model.encode([q], convert_to_numpy=True).astype("float32")

                top_k = 1
                D, I = index.search(q_emb, top_k)
                D = np.array(D[0], dtype=float)
                I = I[0].tolist()

                retrieved = []
                retrieved_meta = []
                distances = []
                
                for idx, d in zip(I, D):
                    if idx is None or idx < 0 or idx >= len(chunks):
                        continue
                    retrieved.append(chunks[idx])
                    retrieved_meta.append(meta[idx])
                    distances.append(float(d))
                
                if not retrieved:
                    response_text = "Irrelevant question. No relevant information found in the file."
                else:
                   
                    max_d = float(np.max(distances)) if len(distances) > 0 else 1.0
                    similarities = [1.0 - (d / (max_d + 1e-9)) for d in distances]
                    avg_sim = float(np.mean(similarities))
                    sim_threshold = 0.45 if len(chunks) > 500 else 0.55

                    if avg_sim < sim_threshold:
                        response_text =  "Irrelevant question. No relevant information found in the file."
                    else:
                        context = "\n".join(retrieved[:3])
                        prompt = f"""
You are a precise assistant. Use ONLY the text in the excerpts below.
Answer in short, clear bullet points (each line = 1 key point).
Do NOT include numbering, document IDs, ISO codes, sections (like 8.4 or 2.1), or page references.
If the answer isn’t clearly stated, reply exactly: "Irrelevant question."

--- Document Excerpts ---
{context}

--- Question ---
{q}

--- Short Answer (bullet points) ---
"""
                        
                        try:
                            res = llm(prompt=prompt, max_tokens=600)
                           
                            if isinstance(res, dict) and "choices" in res and len(res["choices"])>0:
                                answer = res["choices"][0].get("text","").strip()
                            elif isinstance(res, dict) and "content" in res:
                                answer = res.get("content","").strip()
                            else:
                                answer = str(res).strip()
                                answer = deep_clean_answer(answer)
                                if not answer or len(answer.split()) < 3:
                                    answer = "Irrelevant question."
            
                        except Exception as e:
                            answer = f" LLaMA error: {e}"

                        if not re.search(r'[.!?]"?$', answer):
                            for chunk in retrieved:
                                tail = " ".join(answer.split()[-10:])
                                if tail in chunk:
                                    after_tail = chunk.split(tail, 1)[-1]
                                    match = re.search(r'[^.?!]*[.?!]', after_tail)
                                    if match:
                                        answer += match.group(0).strip()
                                    break
                            
                            answer = answer.replace("\n", " ").replace("\r", " ")
                            answer = re.sub(r"\s+", " ", answer).strip()
                                     
                        if len(answer.split()) > 200:
                            answer = " ".join(answer.split()[:200])
                            if not answer.endswith("."):
                                answer += "..."
                                
                        if retrieved_meta:
                            chunk_meta = retrieved_meta[0]
                            page_no = chunk_meta.get("page_no", "?")
                            line_start = chunk_meta.get("line_start", "?")
                            line_end = chunk_meta.get("line_end", "?")
                            response_text = f"{answer}\n\n Reference from file: Page {page_no}, lines {line_start}-{line_end}"
                        else:
                            response_text = answer

                        if not answer:
                            answer = retrieved[0]
                        response_text = answer

                if retrieved:
                    all_bullets = []
                    for chunk in retrieved:
                        bullets = deep_clean_answer(chunk)
                        all_bullets.extend(bullets)                    
                        response_text = "\n".join(all_bullets) if all_bullets else "Irrelevant question. No relevant information found in the file."
                else:
                            response_text = "Irrelevant question. No relevant information found in the file."
      
                # if retrieved:
                #     debug_lines = []
                #     for i, (chunk, sim) in enumerate(zip(retrieved, similarities), start=1):
                #         snippet = re.sub(r"\s+", " ", chunk.strip()) 
                #         match = re.search(r'\.(\s|$)', snippet[150:]) 
                #         if match:
                #             end_pos = 150 + match.start() + 1
                #             snippet = snippet[:end_pos]
                #         else:
                #             snippet = snippet[:200]
                #         debug_lines.append(snippet.strip())
                #     response_text = "\n".join(debug_lines)
                    
                # else:
                #     response_text = "No chunks retrieved."
               
                if response_text and response_text.strip() and response_text.lower() != "irrelevant question.":
                    ChatbotQA.objects.create(
                        user=request.user,
                        uploaded_file=uploaded_file,
                        question=q,
                        answer=response_text.strip()
                        )
               
                
                try:
                    raw = cache.get(redis_key)
                    if raw:
                        chat_history = json.loads(raw)
                    else:
                        chat_history = []

                except Exception:
                    
                      
                    chat_history = []

                    last_chat = chat_history[0] if chat_history else None

            except Exception as exc:
                response_text = f" Error: {exc}\n{traceback.format_exc()}"
                debug_info = ""

    return render(request, "chatbot.html", {
        "response": response_text,
        "debug": debug_info,
        "chat_history": chat_history,
        "last_chat": last_chat,
        "uploaded_file": uploaded_file,
        "reference_items": reference_items
    })

# @csrf_exempt
# def upload_file_api(request):
#     """
#     Safe Upload API:
#     - Works even if user not logged in.
#     - Automatically assigns main user or first CustomUser as fallback.
#     - Builds FAISS index for the uploaded file automatically.
#     """
#     User = get_user_model()

#     try:
       
#         if request.user.is_authenticated and not request.user.is_anonymous:
#             user = request.user
#         else:
            
#             user = User.objects.filter(is_main=True).first() or User.objects.first()
#             if not user:
#                 return JsonResponse({"error": "No default user found. Please create one."}, status=400)

#         if request.method != "POST":
#             return JsonResponse({"error": "Only POST method is allowed"}, status=405)

#         uploaded_file = request.FILES.get("file")
#         category = request.POST.get("category", "General")

#         if not uploaded_file:
#             return JsonResponse({"error": "No file provided"}, status=400)

#         uf = UploadedFile.objects.create(
#             uploaded_by=user,
#             file=uploaded_file,
#             original_name=uploaded_file.name,
#             category=category
#         )

       
#         extracted_text = extract_text_from_pdf_with_fitz(uf.file.path)
#         uf.extracted_text = extracted_text
#         uf.save()

       
#         if user.is_main:
#             owner_id = user.id
#         else:
#             main_user = User.objects.filter(is_main=True).first()
#             owner_id = main_user.id if main_user else user.id

#         index_name = f"user_{owner_id}_file_{uf.id}.index"
#         index_path = os.path.join(FAISS_DIR, index_name)

#         build_faiss_index_from_text(extracted_text, index_path)

      
#         return JsonResponse({
#             "message": " File uploaded successfully",
#             "file_id": uf.id,
#             "uploaded_by": user.username,
#             "category": uf.category
#         })

#     except Exception as e:
#         print("Upload API Error:", traceback.format_exc())
#         return JsonResponse({"error": str(e)}, status=500)
    

# @csrf_exempt
# def chat_api(request):
#     """
#     Chat API:
#     - Uses FAISS for semantic search on a specific file
#     - Automatically picks the right file (if file_id given)
#     - Returns clean JSON with answer + file reference
#     """
#     User = get_user_model()

#     try:
#         if request.method != "POST":
#             return JsonResponse({"error": "Only POST requests are allowed."}, status=405)

#         q = request.POST.get("question", "").strip()
#         file_id = request.POST.get("file_id")

#         if not q:
#             return JsonResponse({"error": "Missing 'question' field."}, status=400)

#         # ✅ Choose correct user (same fallback logic as upload)
#         if request.user.is_authenticated:
#             user = request.user
#         else:
#             user = User.objects.filter(is_main=True).first() or User.objects.first()
#             if not user:
#                 return JsonResponse({"error": "No default user found. Please create one."}, status=400)

#         # ✅ Determine which file to query
#         if file_id:
#             uploaded_file = UploadedFile.objects.filter(id=file_id).first()
#         else:
#             uploaded_file = UploadedFile.objects.filter(uploaded_by=user).order_by("-uploaded_at").first()

#         if not uploaded_file:
#             return JsonResponse({"error": "No uploaded file found."}, status=404)
#         if not uploaded_file.extracted_text:
#             return JsonResponse({"error": "File has no extracted text. Try re-uploading."}, status=400)

#         # ✅ Load or build FAISS index
#         index_name = f"user_{uploaded_file.uploaded_by.id}_file_{uploaded_file.id}.index"
#         index_path = os.path.join(FAISS_DIR, index_name)
#         meta_path = index_path.replace(".index", "_meta.pkl")

#         if not os.path.exists(index_path) or not os.path.exists(meta_path):
#             build_faiss_index_from_text(uploaded_file.extracted_text, index_path)

#         index, chunks, meta = load_faiss_index(index_path)

#         # ✅ Encode and search
#         q_emb = embed_model.encode([q], convert_to_numpy=True).astype("float32")
#         D, I = index.search(q_emb, 3)

#         I = I[0].tolist()
#         D = D[0].tolist()

#         retrieved, retrieved_meta, distances = [], [], []
#         for idx, d in zip(I, D):
#             if idx >= 0 and idx < len(chunks):
#                 retrieved.append(chunks[idx])
#                 retrieved_meta.append(meta[idx])
#                 distances.append(float(d))

#         if not retrieved:
#             return JsonResponse({"answer": "Irrelevant question.", "reference": None, "file_id": uploaded_file.id})

#         # ✅ Similarity threshold check
#         max_d = float(np.max(distances)) if distances else 1.0
#         similarities = [1.0 - (d / (max_d + 1e-9)) for d in distances]
#         avg_sim = float(np.mean(similarities))
#         sim_threshold = 0.45 if len(chunks) > 500 else 0.55

#         if avg_sim < sim_threshold:
#             return JsonResponse({"answer": "Irrelevant question.", "reference": None, "file_id": uploaded_file.id})

#         # ✅ Build prompt for local model
#         context = "\n".join(retrieved[:3])
#         prompt = f"""
# You are a precise assistant. Use ONLY the text below.
# Answer in short bullet points.
# If the question isn’t relevant, reply exactly: "Irrelevant question."

# --- Context ---
# {context}

# --- Question ---
# {q}

# --- Answer ---
# """

#         try:
#             res = llm(prompt=prompt, max_tokens=600)
#             if isinstance(res, dict) and "choices" in res:
#                 answer = res["choices"][0].get("text", "").strip()
#             elif isinstance(res, dict) and "content" in res:
#                 answer = res["content"].strip()
#             else:
#                 answer = str(res).strip()
#         except Exception as e:
#             answer = f"LLaMA error: {e}"

#         answer = deep_clean_answer(answer)

#         # ✅ Add file reference (page, lines)
#         reference = None
#         if retrieved_meta:
#             meta_info = retrieved_meta[0]
#             reference = {
#                 "page_no": meta_info.get("page_no", "?"),
#                 "line_start": meta_info.get("line_start", "?"),
#                 "line_end": meta_info.get("line_end", "?"),
#             }

#         # ✅ Save chat history
#         ChatbotQA.objects.create(
#             user=user,
#             uploaded_file=uploaded_file,
#             question=q,
#             answer=answer
#         )

#         return JsonResponse({
#             "question": q,
#             "answer": answer,
#             "file_id": uploaded_file.id,
#             "file_name": uploaded_file.original_name,
#             "reference": reference,
#             "similarity": round(avg_sim, 3)
#         })

#     except Exception as e:
#         import traceback
#         print("Chat API Error:", traceback.format_exc())
#         return JsonResponse({"error": str(e)}, status=500)






# @csrf_exempt
# def api_chat(request):
#     if request.method == "POST":
#         data = json.loads(request.body)
#         user_message = data.get("message", "").strip()

#         if not user_message:
#             return JsonResponse({"error": "Message is empty"}, status=400)

#         # --- Step 1: Retrieve FAISS index and documents ---
#         index_path = os.path.join(INDEX_PATH, "docs.index")
#         doc_path = os.path.join(INDEX_PATH, "docs.json")

#         if not os.path.exists(index_path) or not os.path.exists(doc_path):
#             return JsonResponse({"error": "No document index found. Please upload files first."}, status=400)

#         index = faiss.read_index(index_path)
#         with open(doc_path, "r", encoding="utf-8") as f:
#             documents = json.load(f)

#         # --- Step 2: Find most relevant context ---
#         query_vec = embedding_model.encode([user_message])
#         scores, indices = index.search(np.array(query_vec).astype("float32"), k=3)

#         context_parts = [documents[i] for i in indices[0] if i < len(documents)]
#         context_text = "\n\n".join(context_parts)

#         # --- Step 3: Combine context + user question ---
#         prompt = f"""
# You are a helpful assistant. Use the following context to answer accurately.

# Context:
# {context_text}

# Question:
# {user_message}

# Answer:
# """

#         # --- Step 4: Send request to local Llama API ---
#         payload = {
#             "model": "mistral-7b-instruct-v0.2",
#             "messages": [
#                 {"role": "system", "content": "You are a helpful and precise assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             "temperature": 0.7,
#             "max_tokens": 400,
#         }

#         try:
#             response = requests.post(LOCAL_LLM_URL, json=payload)
#             result = response.json()
#             bot_reply = result["choices"][0]["message"]["content"].strip()
#         except Exception as e:
#             return JsonResponse({"error": f"LLM error: {str(e)}"}, status=500)

#         # --- Step 5: Store chat history ---
#         ChatbotQA.objects.create(
#             question=user_message,
#             answer=bot_reply
#         )

#         return JsonResponse({"reply": bot_reply})

#     return JsonResponse({"error": "Invalid request method"}, status=405)


# @csrf_exempt
# def api_history(request):
#     """
#     Return chat history from the ChatbotQA table.
#     If the user is logged in, show only their chats.
#     Otherwise, show all (for testing mode).
#     """
#     if request.method == "GET":
      
#         if request.user.is_authenticated:
#             chats = ChatbotQA.objects.filter(user=request.user).order_by("-created_at")
#         else:
#             chats = ChatbotQA.objects.all().order_by("-created_at")[:20]  

       
#         data = [
#             {
#                 "id": c.id,
#                 "user": c.user.username if c.user else "Anonymous",
#                 "question": c.question,
#                 "answer": c.answer,
#                 "created_at": c.created_at.strftime("%Y-%m-%d %H:%M:%S"),
#             }
#             for c in chats
#         ]
#         return JsonResponse({"history": data}, safe=False)
#     else:
#         return JsonResponse({"error": "Only GET method allowed"}, status=405)




