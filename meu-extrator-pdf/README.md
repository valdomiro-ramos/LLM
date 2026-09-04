# 📚 Assistente TDM com RAG (Local)

Sistema de perguntas e respostas sobre manuais técnicos da ferramenta **TDM (Test Data Management)** usando **RAG (Retrieval-Augmented Generation)**, **FAISS** e **Llama 3.2**, rodando 100% local com Docker.

---

## 🧠 O que este projeto faz?

Este sistema permite que você **faça perguntas em linguagem natural** sobre documentação técnica e receba respostas precisas, baseadas no conteúdo dos seus PDFs — tudo **offline**, **sem enviar dados para a nuvem**.

---

## 🛠️ Tecnologias utilizadas

| Tecnologia | Função |
| :--- | :--- |
| **FastAPI** | API para receber perguntas e retornar respostas. |
| **LangChain** | Orquestração do fluxo RAG (busca + geração). |
| **FAISS** | Banco vetorial para busca semântica eficiente. |
| **Ollama + Llama 3.2** | Modelo de linguagem local (offline). |
| **Docker** | Isolamento e portabilidade do ambiente. |

---

## ⚙️ Como funciona o fluxo de perguntas?

1. O usuário envia uma pergunta via API (`/ask`).
2. A pergunta é transformada em um vetor (embedding).
3. O FAISS busca os trechos mais relevantes nos documentos indexados.
4. O LangChain monta um prompt com o contexto encontrado.
5. O Llama 3.2 gera uma resposta precisa com base nesse contexto.

---

## 🚀 Como rodar o projeto

### Pré-requisitos
- Docker Desktop instalado e rodando
- Git (opcional, para clonar)
- ~10 GB de espaço em disco

### Passos

#### 1. Clone o repositório
```bash
git clone https://github.com/valdomiro-ramos/LLM.git
cd LLM/meu-extrator-pdf
```

#### 2. Suba os containers
```bash
docker-compose up -d --build
```

#### 3. Baixe os modelos de IA

> ⚠️ **Importante:** Aguarde o comando anterior terminar completamente antes de executar este passo.

**Se você estiver usando o Prompt de Comando (CMD) ou PowerShell:**
```bash
docker exec -it ollama ollama pull llama3.2
docker exec -it ollama ollama pull nomic-embed-text
```

**Se você estiver usando o Git Bash:**
```bash
winpty docker exec -it ollama ollama pull llama3.2
winpty docker exec -it ollama ollama pull nomic-embed-text
```

#### 4. Verifique se os modelos foram baixados
```bash
docker exec ollama ollama list
```
Deverá aparecer:
```
llama3.2:latest
nomic-embed-text:latest
```

#### 5. Acesse a interface interativa
```
http://localhost:8000/docs
```

---

## 📌 Exemplo de pergunta

Acesse o Swagger (`http://localhost:8000/docs`), clique em `POST /ask` e envie:

```json
{
  "pergunta": "Quais bancos de dados o TDM pode conectar?"
}
```

**Resposta esperada:**
> *O TDM pode conectar a bancos de dados como Oracle, DB2, SQL Server e PostgreSQL, conforme documentação técnica.*

---

## 🗂️ Estrutura do projeto

```
LLM/
└── meu-extrator-pdf/
    ├── app/
    │   ├── main.py                # Código principal da API
    │   ├── requirements.txt       # Dependências Python
    │   ├── Dockerfile             # Configuração do container
    │   └── pdfs/                  # Coloque seus PDFs aqui
    ├── faiss_index/               # Índice vetorial (criado automaticamente)
    ├── docker-compose.yml         # Orquestração dos containers
    └── README.md                  # Este arquivo
```

---

## 🔐 Privacidade e segurança

- **100% offline:** Nenhum dado sai do seu computador.
- **Totalmente local:** Modelos, índices e PDFs rodam exclusivamente na sua máquina.
- **Sem APIs externas:** Tudo roda dentro do Docker, sem dependência de serviços em nuvem.

---

## 📈 Próximos passos (melhorias futuras)

- [ ] Adicionar metadados (categoria, versão, autor) via CSV.
- [ ] Suporte a múltiplos formatos (PDF, DOCX, TXT, CSV).
- [ ] Interface web amigável (Streamlit ou React).
- [ ] Filtros por data, categoria e versão.
- [ ] Exportação de respostas para PDF ou Excel.

---

## 👨‍💻 Autor

Feito com ☕ e 🧠 por [Valdomiro Ramos] – [GitHub](https://github.com/valdomiro-ramos)

---

## 📄 Licença

Este projeto é de uso livre para fins educacionais e de estudo.

---

## ⭐ Créditos

- [LangChain](https://www.langchain.com/)
- [FAISS](https://faiss.ai/)
- [Ollama](https://ollama.com/)
- [Llama 3.2](https://ai.meta.com/llama/)