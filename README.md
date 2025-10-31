1️⃣ Clone Repository
git clone https://github.com/yourusername/chatdoc.git
cd chatdoc

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate     # (on Windows)

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Setup NLTK Data

If not already downloaded:

python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
Or manually include:

D:/Python projects/pdf_chatbot/pdf_chat/nltk_data

5️⃣ Model Setup

Place your LLaMA model file (llama-2-7b-chat.Q4_K_M.gguf) in:

D:/Python projects/pdf_chatbot/models/


and update this path in your views.py:

llm = Llama(
    model_path="D:/Python projects/pdf_chatbot/models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=8192,
    n_threads=8,
)


 How It Works

1️⃣ User Uploads File
→ Extracted text is cleaned and chunked.
→ FAISS builds an index with embeddings.

2️⃣ User Asks Question
→ Query is embedded and matched against FAISS index.
→ Top relevant chunks (context) are retrieved.

3️⃣ Prompt Sent to LLaMA
→ The context + question are fed into the LLaMA model.
→ The model generates a concise, contextually grounded answer.

4️⃣ Response Displayed
→ Answer is cleaned, formatted, and displayed with references.
→ Stored in DB and Redis for chat history.