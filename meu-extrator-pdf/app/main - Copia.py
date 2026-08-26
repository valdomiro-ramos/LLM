import os
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Força a desativação da telemetria (garantia extra)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

# --- 1. Configurações ---
PDF_DIR = "pdfs"          # Pasta onde estão os extratos
FAISS_INDEX_DIR = "faiss_index"  # Pasta para salvar o índice FAISS
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Modelos que vamos usar (baixe eles no Ollama)
EMBEDDING_MODEL = "nomic-embed-text"  # Modelo para criar vetores
LLM_MODEL = "llama3.2"               # Modelo para gerar respostas

# --- 2. Inicialização do FastAPI ---
app = FastAPI(
    title="Extrator de Extratos com IA",
    description="Faça perguntas sobre seus PDFs de extrato",
    version="1.0"
)

# --- 3. Função para carregar e indexar os PDFs ---
def load_and_index_pdfs():
    print("🔍 Procurando PDFs na pasta:", PDF_DIR)
    
    # Lista todos os PDFs na pasta
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if not pdf_files:
        print("⚠️ Nenhum PDF encontrado! Coloque seus arquivos na pasta 'pdfs'.")
        return None
    
    print(f"📄 Encontrados {len(pdf_files)} PDF(s). Carregando...")
    
    all_documents = []
    for pdf_path in pdf_files:
        print(f"  -> Carregando: {os.path.basename(pdf_path)}")
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            print(f"  ❌ Erro ao carregar {pdf_path}: {e}")
    
    if not all_documents:
        print("❌ Nenhum texto foi extraído dos PDFs.")
        return None
    
    print(f"✅ {len(all_documents)} páginas extraídas. Dividindo em pedaços...")
    
    # Divide o texto em chunks menores para buscar com mais precisão
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # Tamanho de cada pedaço
        chunk_overlap=150,     # Sobreposição para não perder contexto
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"✂️ Gerados {len(chunks)} pedaços (chunks).")
    
    # Conecta no Ollama para gerar os embeddings (vetores)
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )
    
    # Cria o índice FAISS (NÃO usa ChromaDB, então NÃO tem erro de telemetria!)
    print(f"💾 Salvando vetores no FAISS (pasta: {FAISS_INDEX_DIR})...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Salva o índice no disco para reutilizar depois
    vectorstore.save_local(FAISS_INDEX_DIR)
    print("✅ Indexação concluída com sucesso!")
    return vectorstore

# --- 4. Carrega/Indexa os PDFs na inicialização ---
# Tenta carregar um índice existente primeiro (para não precisar recriar toda vez)
vectorstore = None
if os.path.exists(FAISS_INDEX_DIR):
    print("📂 Índice FAISS existente encontrado. Carregando...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
        vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        print("✅ Índice FAISS carregado com sucesso!")
    except Exception as e:
        print(f"⚠️ Erro ao carregar índice existente: {e}. Recriando...")
        vectorstore = None

# Se não conseguiu carregar, cria do zero
if vectorstore is None:
    vectorstore = load_and_index_pdfs()

# Se conseguiu indexar, cria a chain de perguntas e respostas
qa_chain = None
if vectorstore:
    print("🧠 Configurando o modelo de linguagem...")
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.3,  # Menos criativo, mais preciso para fatos
        num_predict=512   # Limite de tokens na resposta
    )
    
    # Cria a corrente de RAG (Recuperação + Geração)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Busca os 4 pedaços mais relevantes
        ),
        return_source_documents=False
    )
    print("🚀 Sistema pronto para perguntas!")
else:
    print("❌ Sistema iniciado sem indexação. Coloque PDFs na pasta 'pdfs' e reinicie.")

# --- 5. Modelo de dados para a requisição ---
class Question(BaseModel):
    pergunta: str

# --- 6. Endpoint da API ---
@app.post("/ask")
async def ask_question(question: Question):
    if not qa_chain:
        raise HTTPException(
            status_code=503, 
            detail="Sistema não inicializado. Certifique-se de que há PDFs na pasta 'pdfs' e que o Ollama está rodando."
        )
    
    try:
        resposta = qa_chain.run(question.pergunta)
        return {"pergunta": question.pergunta, "resposta": resposta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")

# --- 7. Endpoint de saúde (opcional) ---
@app.get("/")
async def root():
    return {"status": "online", "mensagem": "Envie uma pergunta para /ask com JSON: {'pergunta': 'texto'}"}

# --- 8. Endpoint para recarregar os PDFs (útil se adicionar novos arquivos) ---
@app.post("/reload")
async def reload_pdfs():
    global vectorstore, qa_chain
    # Remove o índice antigo (opcional)
    import shutil
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
    
    vectorstore = load_and_index_pdfs()
    if vectorstore:
        llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
        )
        return {"status": "success", "message": "PDFs recarregados com sucesso!"}
    else:
        qa_chain = None
        return {"status": "error", "message": "Nenhum PDF encontrado."}