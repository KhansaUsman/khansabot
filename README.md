# KhansaBot 🤖

## Student Information
- **Student Name:** Khansa Usman
- **Student ID:** [f2022065303]
- **Project Title:** KhansaBot — Personal RAG Chatbot
- **Chatbot Name:** KhansaBot

## Project Description
KhansaBot is a Retrieval-Augmented Generation (RAG) based personal chatbot built using LangChain, FAISS, Groq (LLaMA 3.3 70B), HuggingFace sentence-transformers, and Streamlit. It answers questions about Khansa Usman using only her personal documents — CV, FYP details, and current semester coursework — without hallucinating or guessing information not present in the data.


## Tech Stack
| Component | Technology |
|---|---|
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local, free) |
| Vector Store | FAISS |
| Framework | LangChain |
| Interface | Streamlit |

## Project Structure
khansabot/

├── app.py              # Streamlit chat interface

├── rag_pipeline.py     # Core RAG logic

├── requirements.txt    # Python dependencies

├── .env.example        # Environment variable template

├── .gitignore          # Git ignore rules

├── data/               # Personal dataset

│   ├── Khansa_Usman_Resume-1.pdf

│   ├── fyp_details.txt

│   └── courses_semester.txt

└── README.md
## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/KhansaUsman/khansabot.git
cd khansabot
```

### 2. Create and activate virtual environment
```bash
python -m venv venv --without-pip
venv\Scripts\activate
python -m ensurepip --upgrade
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Add your GROQ_API_KEY to the .env file
```

### 5. Run the app
```bash
streamlit run app.py
```

## Dataset
- **Khansa_Usman_Resume-1.pdf** — Personal CV
- **fyp_details.txt** — Final Year Project details (AI-Enhanced IoT Security System)
- **courses_semester.txt** — Current semester courses at UMT

## Features
- ✅ Personal RAG pipeline with grounded answers
- ✅ Anti-hallucination guardrails
- ✅ Conversation history maintenance
- ✅ Multi-document retrieval (CV + FYP + courses)
- ✅ Refuses to answer questions not in the dataset
- ✅ Streamlit chat interface with KhansaBot identity