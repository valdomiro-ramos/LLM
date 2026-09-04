import os
import glob
import shutil
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Força a desativação da telemetria ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

# --- 1. Configurações ---
PDF_DIR = "pdfs"
FAISS_INDEX_DIR = "faiss_index"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"  # Pode trocar por "mistral" ou "phi3" se quiser

# --- 2. Inicialização do FastAPI ---
app = FastAPI(
    title="Assistente TDM com IA",
    description="Faça perguntas sobre a ferramenta TDM (Test Data Management)",
    version="1.0"
)

# --- 3. Função para carregar e indexar os PDFs ---
def load_and_index_pdfs():
    print("🔍 Procurando PDFs na pasta:", PDF_DIR)
    
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
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"✂️ Gerados {len(chunks)} pedaços (chunks).")
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )
    
    print(f"💾 Salvando vetores no FAISS (pasta: {FAISS_INDEX_DIR})...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    vectorstore.save_local(FAISS_INDEX_DIR)
    
    print("✅ Indexação concluída com sucesso!")
    return vectorstore

# --- 4. Carrega ou cria o índice FAISS ---
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

if vectorstore is None:
    vectorstore = load_and_index_pdfs()

# --- 5. Configura a chain com prompt personalizado ---
qa_chain = None
if vectorstore:
    print("🧠 Configurando o modelo de linguagem...")
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.2,
        num_predict=512
    )

    template = """
    Você é um especialista técnico em manuais de software.
    Responda perguntas sobre a ferramenta TDM (Test Data Management) com base APENAS no contexto fornecido.
    Forneça respostas técnicas, claras, objetivas e em português.
    Se a informação não estiver no contexto, responda: "Não encontrei essa informação na documentação fornecida."
    Se for uma lista, formate em tópicos numerados.
    Se a pergunta pedir um passo a passo, descreva o processo de forma lógica.

    Contexto:
    {context}

    Pergunta: {question}
    Resposta:"""
    
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 15}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("🚀 Sistema pronto para perguntas!")
else:
    print("❌ Sistema iniciado sem indexação. Coloque PDFs na pasta 'pdfs' e reinicie.")

# --- 6. Modelo de dados para a requisição ---
class Question(BaseModel):
    pergunta: str

# --- 7. Endpoint da API ---
@app.post("/ask")
async def ask_question(question: Question):
    if not qa_chain:
        raise HTTPException(503, detail="Sistema não inicializado.")
    
    pergunta_original = question.pergunta
    
    # 1. Extrai o ano (se houver)
    ano_match = re.search(r'\b(20\d{2})\b', pergunta_original)
    ano = ano_match.group() if ano_match else None
    
    try:
        docs = vectorstore.similarity_search(pergunta_original, k=20)
        
        # 2. Se tiver ano, FILTRA os chunks pela fonte
        if ano:
            chunks_filtrados = []
            for doc in docs:
                fonte = doc.metadata.get("source", "")
                if ano in fonte:
                    chunks_filtrados.append(doc)
            
            if not chunks_filtrados:
                return {
                    "pergunta": pergunta_original,
                    "resposta": f"Não encontrei documentos do ano {ano} na base.",
                    "filtro_aplicado": f"ano {ano} (nenhum chunk encontrado)"
                }
            
            context = "\n\n".join([doc.page_content for doc in chunks_filtrados])
            
            llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.2)
            prompt_final = f"""
            Você é um especialista técnico em manuais de software.
            Responda perguntas sobre a ferramenta TDM com base APENAS no contexto abaixo.
            Forneça respostas técnicas, claras e objetivas.
            Se a informação não estiver no contexto, responda: "Não encontrei essa informação na documentação fornecida."

            Contexto:
            {context}
            
            Pergunta: {pergunta_original}
            Resposta:"""
            
            resposta = llm.invoke(prompt_final)
            
            return {
                "pergunta": pergunta_original,
                "resposta": resposta,
                "filtro_aplicado": f"ano {ano} (filtrou {len(chunks_filtrados)} chunks)"
            }
        
        # Se não tiver ano, usa a chain normal
        else:
            resposta = qa_chain.run(pergunta_original)
            return {"pergunta": pergunta_original, "resposta": resposta, "filtro_aplicado": None}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")

# --- 8. Endpoint de saúde ---
@app.get("/")
async def root():
    return {"status": "online", "mensagem": "Assistente TDM pronto para perguntas."}

# --- 9. Endpoint para recarregar os PDFs ---
@app.post("/reload")
async def reload_pdfs():
    global vectorstore, qa_chain
    
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
        print("🗑️ Índice FAISS antigo removido.")
    
    vectorstore = load_and_index_pdfs()
    if vectorstore:
        llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.2)
        
        template = """
        Você é um especialista técnico em manuais de software.
        Responda perguntas sobre a ferramenta TDM com base APENAS no contexto fornecido.
        Forneça respostas técnicas, claras e objetivas.
        Se a informação não estiver no contexto, responda: "Não encontrei essa informação na documentação fornecida."

        Contexto:
        {context}

        Pergunta: {question}
        Resposta:"""
        
        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 15}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return {"status": "success", "message": "Documentos recarregados com sucesso!"}
    else:
        qa_chain = None
        return {"status": "error", "message": "Nenhum PDF encontrado."}

# --- 10. Endpoint de Debug ---
@app.post("/debug/search")
async def debug_search(question: Question):
    if not vectorstore:
        raise HTTPException(503, detail="Vectorstore não inicializado.")
    
    docs = vectorstore.similarity_search(question.pergunta, k=15)
    
    return {
        "pergunta": question.pergunta,
        "total_chunks_recuperados": len(docs),
        "chunks": [
            {
                "fonte": d.metadata.get("source", "N/A"),
                "pagina": d.metadata.get("page", "N/A"),
                "texto": d.page_content[:500] + "..." if len(d.page_content) > 500 else d.page_content
            }
            for d in docs
        ]
    }