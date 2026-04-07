import os
import time
from dotenv import load_dotenv

# Loader PDF
from langchain_community.document_loaders import PyPDFLoader

# Splitter de texto
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# LLM Google/Gemini para LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Banco vetorial
from langchain_community.vectorstores import Chroma

# Cadeia RAG
from langchain_classic.chains import RetrievalQA



# 1. Carregar as variáveis de ambiente do arquivo .env

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("A variável GEMINI_API_KEY não foi encontrada no arquivo .env")



# 2. Carrega os PDFs e adiciona metadados de medicamento e página
caminhos_bulas = [
    "bulaDipirona.pdf",
    "bulaParacetamol.pdf"
]

documentos = []

for caminho in caminhos_bulas:
    loader = PyPDFLoader(caminho)
    docs = loader.load()

    for doc in docs:
        doc.metadata["medicamento"] = os.path.basename(caminho).replace(".pdf", "")

    documentos.extend(docs)

print(f"Quantidade de páginas carregadas: {len(documentos)}")



# 3. Divide os documentos em chunks menores, mantendo metadados

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # menos chunks que 600
    chunk_overlap=100  # overlap menor
)

chunks = text_splitter.split_documents(documentos)

print(f"Quantidade total de chunks gerados: {len(chunks)}")



# 4. Rotula os chunks com categorias baseadas em palavras-chave, mantendo os metadados que já existem

for chunk in chunks:
    texto = chunk.page_content.lower()

    if "composição" in texto:
        chunk.metadata["categoria"] = "composicao"

    elif "indicação" in texto or "para que este medicamento é indicado" in texto:
        chunk.metadata["categoria"] = "indicacao"

    elif "como este medicamento funciona" in texto or "ação" in texto:
        chunk.metadata["categoria"] = "como_funciona"

    elif "contraindicação" in texto or "quando não devo usar" in texto:
        chunk.metadata["categoria"] = "contraindicacao"

    elif "advertência" in texto or "precaução" in texto or "o que devo saber antes de usar" in texto:
        chunk.metadata["categoria"] = "advertencias_precaucoes"

    elif "interação" in texto or "interações medicamentosas" in texto:
        chunk.metadata["categoria"] = "interacoes"

    elif "dose" in texto or "posologia" in texto or "modo de usar" in texto:
        chunk.metadata["categoria"] = "posologia_modo_uso"

    elif "reações adversas" in texto or "quais os males" in texto:
        chunk.metadata["categoria"] = "reacoes_adversas"

    elif "onde, como e por quanto tempo posso guardar" in texto or "armazenar" in texto:
        chunk.metadata["categoria"] = "armazenar"

    elif "quantidade maior do que a indicada" in texto or "superdosagem" in texto:
        chunk.metadata["categoria"] = "superdosagem"

    else:
        chunk.metadata["categoria"] = "geral"



# Verificação de categorias atribuídas

#chunks_aleatorios = random.sample(chunks, 2)

#for i, chunk in enumerate(chunks_aleatorios, start=1):
    #print(f"Chunk {i+1}:")
    #print(f"Metadados: {chunk.metadata}")
    #print(f"\nConteúdo (ínicio): {chunk.page_content[:300]}")



# 5. Embeddings com google gemini (Free tier)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)



# 6. Cria o banco vetorial chorma localmente, utilizando os embeddings do Gemini e persistindo em disco para evitar perda de dados

persist_directory = "./chroma_bulas"
collection_name = "chroma_bulas"

vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_directory
)



# 7. Insere os chunks no banco vetorial em lotes. A pausa é para respeitar o free tier e não dar erro no google embeddings

batch_size = 10
pause_seconds = 65

print("Iniciando inserção em lotes no Chroma...")

for i in range(0, len(chunks), batch_size):
    lote = chunks[i:i + batch_size]
    numero_lote = i // batch_size + 1
    total_lotes = (len(chunks) + batch_size - 1) // batch_size

    print(f"Adicionando lote {numero_lote}/{total_lotes} com {len(lote)} chunks...")
    vectorstore.add_documents(lote)

    if i + batch_size < len(chunks):
        print(f"Aguardando {pause_seconds}s para respeitar o free tier...")
        time.sleep(pause_seconds)

print("Base vetorial criada com sucesso.")



# 8. Criação do retriever

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# 9. Configuração do LLM Gemini para LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)



# 10. Cadeia RAG

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# 11. Pergunta teste

pergunta = "Quais as contraindicações do medicamento Dipirona?"

try:
    resposta = qa_chain.invoke({"query": pergunta})

    print("\n=== RESPOSTA DO AGENTE ===")
    print(resposta["result"])

    print("\n=== TRECHOS UTILIZADOS COMO CONTEXTO ===")
    for i, doc in enumerate(resposta["source_documents"], start=1):
        print(f"\n---- Trecho {i} ----")
        print(f"Medicamento: {doc.metadata.get('medicamento', 'N/A')}")
        print(f"Categoria: {doc.metadata.get('categoria', 'N/A')}")
        print(f"Documento: {doc.metadata.get('source', 'Documento desconhecido')}")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print("\nConteúdo do chunk:")
        print(doc.page_content[:800])
        print("\n-------------------")

except Exception as e:
    print("\nErro ao consultar o Gemini:")
    print(e)