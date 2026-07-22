# Intelligent_Customer_Support_System_RAG
An AI-powered Intelligent Customer Support System built using Retrieval-Augmented Generation (RAG). Users can upload PDF documents, extract and semantically search their content using Sentence Transformers and FAISS, and receive context-aware answers through the Llama 3.1:8b LLM running locally with Ollama. Built with Python and FastAPI.
Here is a complete **README.md** you can directly copy into your GitHub repository.

#  Intelligent Customer Support System using RAG

An AI-powered customer support system that uses **Retrieval-Augmented Generation (RAG)** to answer questions based on uploaded PDF documents. The system extracts content from PDFs, converts the content into semantic vector embeddings, retrieves the most relevant information using FAISS, and generates context-aware answers using the Llama 3.1:8b Large Language Model through Ollama.

---

##  Project Overview

Traditional customer support systems often depend on manually searching through product manuals, warranty documents, and technical guides. This process can be time-consuming and inefficient.

This project provides an intelligent solution where users can upload product-related PDF documents and ask questions in natural language. The system searches the uploaded documents based on semantic meaning and generates answers using retrieved document context.

The system follows the **Retrieval-Augmented Generation (RAG)** architecture.

```text
User Question
      ↓
Question Embedding
      ↓
Semantic Search using FAISS
      ↓
Retrieve Relevant Document Chunks
      ↓
Build Context
      ↓
Llama 3.1:8b using Ollama
      ↓
AI-Generated Answer
```

---

##  Features

*  Upload PDF documents
*  Extract text from PDF files
*  Split documents into smaller text chunks
*  Generate semantic embeddings
*  Fast vector similarity search using FAISS
*  AI-powered question answering
*  Local LLM support using Llama 3.1:8b and Ollama
*  Local document processing and data privacy
*  FastAPI-based backend
*  PDF viewing support
*  Natural language interaction

---

## 🛠️ Technologies Used

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| Python                | Core programming language |
| FastAPI               | Backend API framework     |
| HTML                  | User interface            |
| PDFPlumber            | PDF text extraction       |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2      | Embedding model           |
| FAISS                 | Vector similarity search  |
| NumPy                 | Numerical operations      |
| Pickle                | Storing text chunks       |
| Ollama                | Local LLM execution       |
| Llama 3.1 8B          | Language model            |
| Uvicorn               | ASGI server               |

---

##  How RAG Works in This Project

Retrieval-Augmented Generation combines two major processes:

### 1. Retrieval

Relevant information is retrieved from the uploaded document.

```text
PDF
 ↓
Extract Text
 ↓
Split into Chunks
 ↓
Generate Embeddings
 ↓
Store in FAISS
```

### 2. Generation

The retrieved information is sent to the Llama 3.1:8b model.

```text
User Question
      ↓
Retrieve Relevant Context
      ↓
Send Context + Question to LLM
      ↓
Generate Answer
```

This allows the AI to answer questions based on the uploaded documents.

---

##  System Architecture

```text
┌────────────────────┐
│       User         │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   FastAPI Backend  │
└───────┬─────┬──────┘
        │     │
        │     └───────────────────────┐
        ▼                             ▼
┌───────────────┐              ┌──────────────┐
│ PDF Upload    │              │ User Query   │
└───────┬───────┘              └──────┬───────┘
        │                             │
        ▼                             ▼
┌───────────────┐              ┌──────────────┐
│ PDFPlumber    │              │ Embedding    │
│ Text Extract  │              │ Generation   │
└───────┬───────┘              └──────┬───────┘
        │                             │
        ▼                             ▼
┌───────────────┐              ┌──────────────┐
│ Text Chunks   │              │ FAISS Search │
└───────┬───────┘              └──────┬───────┘
        │                             │
        ▼                             ▼
┌───────────────┐              ┌──────────────┐
│ Embeddings    │              │ Top Relevant │
│               │              │ Chunks       │
└───────┬───────┘              └──────┬───────┘
        │                             │
        ▼                             ▼
┌──────────────────────────────────────────────┐
│             Context + User Question          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Ollama + LLM   │
              │  Llama 3.1 8B   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Final Answer   │
              └─────────────────┘
```

---

## 📂 Project Structure

```text
Intelligent-Customer-Support-RAG/
│
├── main.py
├── index.html
├── requirements.txt
├── README.md
│
├── uploads/
│   └── Uploaded PDF files
│
└── vectors/
    ├── chunks.pkl
    └── vectors.index
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd Intelligent-Customer-Support-RAG
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

---

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
fastapi
uvicorn
python-multipart
pdfplumber
sentence-transformers
faiss-cpu
numpy
ollama
```

---

##  Install and Configure Ollama

Install Ollama on your system and download the Llama 3.2 model:

```bash
ollama pull llama3.1:8b
```

Start Ollama:

```bash
ollama serve
```

The Python application uses:

```python
ollama.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
```

---

##  Run the Application

Start the FastAPI server:

```bash
python main.py
```

Or:

```bash
uvicorn main:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

---

##  Application Workflow

### Step 1: Upload PDF

The user uploads a product manual or warranty document.

### Step 2: Extract Text

Text is extracted from the PDF using `pdfplumber`.

### Step 3: Chunk the Text

The extracted text is split into smaller overlapping chunks.

```python
chunk_size = 500
overlap = 50
```

### Step 4: Generate Embeddings

Each text chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

### Step 5: Store Embeddings

The vectors are stored in a FAISS index.

```text
vectors/vectors.index
```

The text chunks are stored using Pickle.

```text
vectors/chunks.pkl
```

### Step 6: Ask a Question

The user asks a natural-language question.

### Step 7: Retrieve Relevant Information

FAISS retrieves the top three semantically similar text chunks.

### Step 8: Generate Answer

The retrieved context is sent to Llama 3.1:8b through Ollama.

---

##  API Endpoints

### Home Page

```text
GET /
```

Returns the HTML user interface.

---

### Upload PDF

```text
POST /upload
```

Uploads and indexes a PDF document.

#### Request

```text
multipart/form-data
```

#### Response

```json
{
  "filename": "manual.pdf",
  "status": "Indexed Successfully"
}
```

---

### Ask a Question

```text
POST /ask
```

Accepts a natural-language question.

#### Request

```text
question=What is the warranty period?
```

#### Response

```json
{
  "answer": "The warranty period is two years."
}
```

---

### View PDF

```text
GET /view-pdf/{filename}
```

Displays an uploaded PDF file in the browser.

---

##  Semantic Search

The system does not rely only on exact keyword matching.

For example:

```text
Question:
How long is the battery coverage?
```

The document may contain:

```text
The battery is covered under warranty for two years.
```

Even though the words are different, the system understands that both sentences have similar meanings.

This is achieved using:

```text
Sentence Transformers
        +
FAISS
```

---

##  Core RAG Components

### Sentence Transformers

Converts text into numerical vector embeddings.

```python
model.encode(text)
```

---

### FAISS

Performs similarity search between the question vector and document vectors.

```python
index.search(query_vector, k=3)
```

---

### Ollama

Runs the Llama 3.2 model locally.

```python
ollama.chat(...)
```

---

##  Example Usage

### User Question

```text
What is the warranty period for the battery?
```

### Retrieved Context

```text
The laptop battery is covered under warranty for two years.
```

### AI Response

```text
The battery warranty period is two years.
```

---

##  Security and Privacy

The system supports local document processing using Ollama. Uploaded documents and questions can remain within the local environment, reducing the need to send sensitive documents to external cloud AI services.

For production deployment, additional security features should be added, including:

* File type validation
* File size restrictions
* Authentication
* Authorization
* Secure file storage
* Input sanitization
* Rate limiting

---

##  Current Limitations

* Supports primarily text-based PDFs.
* Scanned PDFs may require OCR.
* The current implementation stores one active FAISS index.
* Authentication is not currently implemented.
* The system uses local file storage.
* Large documents may require improved chunking strategies.
* LLM response quality depends on the retrieved context.

---

##  Use Cases

This system can be used for:

* Product manual assistance
* Warranty support
* Technical troubleshooting
* Company knowledge bases
* Educational document Q&A
* Customer service automation
* Internal enterprise document search
* FAQ automation

---

##  Project Benefits

* Reduces manual document searching.
* Provides faster customer support.
* Uses semantic understanding instead of simple keyword matching.
* Reduces dependence on external cloud AI APIs.
* Improves privacy through local processing.
* Can be extended to multiple business domains.

---

##  Author

**Kishore Kumar S**

Computer Science and Engineering (Internet of Things)

---

## 📄 License

This project is developed for educational and academic purposes.

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ star.
