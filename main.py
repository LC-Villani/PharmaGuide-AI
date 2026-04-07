# import os
# from dotenv import load_dotenv

# # Loader PDF
# from langchain_community.document_loaders import PyPDFLoader

# # Splitter de texto
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Embeddings Google
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# # LLM Google/Gemini para LangChain
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Banco vetorial
# from langchain_community.vectorstores import Chroma

# # Cadeia RAG
# from langchain_classic.chains import RetrievalQA

# from google import genai

# import random

# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# from langchain_community.vectorstores import Chroma

# load_dotenv()


# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# caminhos_bulas = [
#     "bulaDipirona.pdf",
#     "bulaParacetamol.pdf"
# ]

# documentos = []

# for caminho in caminhos_bulas:
#     loader = PyPDFLoader(caminho)
#     docs = loader.load()

#     for doc in docs:
#         doc.metadata["medicamento"] = caminho.split("/")[-1].replace(".pdf", "")

#     documentos.extend(docs)

# #len(documentos)

# #print(len(documentos))


# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=600, # Tamanho máximo de cada chunk
#     chunk_overlap=150 # Sobreposição entre chunks para manter o contexto
#     )

# chunks = text_splitter.split_documents(documentos)

# # len(chunks)

# #print(len(chunks))


# for chunk in chunks: 

#     texto = chunk.page_content.lower()

#     if "indicação de medicament" in texto or "composição" in texto: 
#         chunk.metadata["categoria"] = "indicacao"
    
#     elif "indicação" in texto or "para que este medicamento é indicado" in texto: 
#         chunk.metadata["categoria"] = "indicacao"

#     elif "como este medicamento funciona" in texto or "ação" in texto: 
#         chunk.metadata["categoria"] = "como_funciona"

#     elif "contraindicação" in texto or "quando não devo usar" in texto:
#         chunk.metadata["categoria"] = "contraindicacao"

#     elif "advertência" in texto or "precaução" in texto or "o que devo saber antes de usar" in texto:
#         chunk.metadata["categoria"] = "advertencias_precaucoes"
    
#     elif "interação" in texto or "interações medicamentosas" in texto:
#         chunk.metadata["categoria"] = "interacoes"

#     elif "dose" in texto or "posologia" in texto or "o que devo saber antes de usar" in texto:
#         chunk.metadata["categoria"] = "posologia_modo_uso"
    
#     elif "reações adversas" in texto or "quais os males" in texto:
#         chunk.metadata["categoria"] = "reacoes_adversas"

#     elif "onde, como e por quanto tempo posso guardar" in texto or "armazenar" in texto:
#         chunk.metadata["categoria"] = "armazenar"
    
#     elif "quantidade maior do que a indicada" in texto or "superdosagem" in texto:
#         chunk.metadata["categoria"] = "superdosagem"

#     else: 
#         chunk.metadata["categoria"] = "geral"


# #chunks_aleatorios = random.sample(chunks, 2)

# #for i, chunk in enumerate(chunks_aleatorios, start=1):
#     #print(f"Chunk {i+1}:")
#     #print(f"Metadados: {chunk.metadata}")
#     #print(f"\nConteúdo (ínicio): {chunk.page_content[:300]}")

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001"
#     )


# chunks_teste = chunks[:10]

# vectorstore = Chroma.from_documents(
#     documents=chunks_teste,
#     embedding=embeddings,
#     collection_name="chroma_bulas",
#     persist_directory="./chroma_bulas"
# )


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


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("A variável GEMINI_API_KEY não foi encontrada no .env")


# =========================
# 1. CARREGAR PDFs
# =========================
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

print(f"Quantidade de páginas/documentos carregados: {len(documentos)}")


# =========================
# 2. DIVIDIR EM CHUNKS
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # maior para gerar menos chunks
    chunk_overlap=100  # overlap menor para reduzir volume
)

chunks = text_splitter.split_documents(documentos)

print(f"Quantidade total de chunks gerados: {len(chunks)}")


# =========================
# 3. CATEGORIZAR CHUNKS
# =========================
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


# =========================
# 4. EMBEDDINGS GOOGLE
# =========================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)


# =========================
# 5. CRIAR / CARREGAR CHROMA
# =========================
persist_directory = "./chroma_bulas"
collection_name = "chroma_bulas"

vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_directory
)


# =========================
# 6. INSERIR EM LOTES (FREE TIER)
# =========================
batch_size = 10      # lote pequeno
pause_seconds = 65   # espera segura para não bater no limite

# Se quiser testar primeiro, limite aqui:
# chunks = chunks[:10]

print("Iniciando inserção em lotes no Chroma...")

for i in range(0, len(chunks), batch_size):
    lote = chunks[i:i + batch_size]

    print(f"Adicionando lote {i // batch_size + 1} com {len(lote)} chunks...")
    vectorstore.add_documents(lote)

    # Só espera se ainda houver mais lotes
    if i + batch_size < len(chunks):
        print(f"Aguardando {pause_seconds}s para respeitar o free tier...")
        time.sleep(pause_seconds)

print("Base vetorial criada com sucesso.")


# =========================
# 7. CRIAR RETRIEVER
# =========================
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# =========================
# 8. LLM GEMINI
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)


# =========================
# 9. CADEIA RAG
# =========================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# =========================
# 10. TESTE DE PERGUNTA
# =========================
pergunta = "Para que a dipirona é indicada?"

resposta = qa_chain.invoke({"query": pergunta})

print("\n=== RESPOSTA ===")
print(resposta["result"])

print("\n=== FONTES ===")
for i, doc in enumerate(resposta["source_documents"], start=1):
    print(f"\nFonte {i}:")
    print(doc.metadata)
    print(doc.page_content[:300])



