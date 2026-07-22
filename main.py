import os
import pickle
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
import pdfplumber  
from sentence_transformers import SentenceTransformer
import ollama

app = FastAPI()

# Create folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("vectors", exist_ok=True)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
CHUNKS_FILE = "vectors/chunks.pkl"
INDEX_FILE = "vectors/vectors.index"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 1. Text Extraction
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return {"status": "Error", "message": f"Could not read PDF: {str(e)}"}

    if len(text.strip()) < 50:
        return {"status": "Error", "message": "Failed: No text could be extracted. The PDF might be a scanned image."}

    # 2. Chunking
    words = text.split()
    chunks = []
    chunk_size = 500
    overlap = 50
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    # 3. Create Vectors
    embeddings = model.encode(chunks).astype('float32')
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    # 4. Save to FAISS
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    return {"filename": file.filename, "status": "Indexed Successfully!", "count": len(chunks)}

@app.post("/ask")
async def ask(question: str = Form(...)):
    if not os.path.exists(INDEX_FILE):
        return {"answer": "I don't have any data yet. Please upload a PDF!"}

    # Load data
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    # Search top 3 chunks
    query_vec = model.encode([question]).astype('float32')
    distances, indices = index.search(query_vec, k=3)
    
    context = ""
    for i in indices[0]:
        if i != -1 and i < len(chunks):
            context += chunks[i] + "\n"

    # Ollama Chat
    prompt = f"""
    You are a Customer Support Assistant. Use the manual context below to answer the user's question.
    If the answer isn't there, say you don't know.

    Context:
    {context}

    Question: 
    {question}
    """
    
    try:
        response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
        return {"answer": response["message"]["content"]}
    except Exception as e:
        return {"answer": "Ollama is not responding. Make sure it is running locally."}

@app.get("/view-pdf/{filename}")
async def view_pdf(filename: str):
    return FileResponse(path=f"uploads/{filename}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)