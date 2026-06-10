import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

print("=== INICIANDO PIPELINE RAG REAL (CON PDF) ===")

# --- PASO 1: Ingesta (PyPDFLoader) ---
print("\n[Paso 1] Cargando archivo PDF y fragmentando...")
pdf_path = "documento_reglas_embol.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"No se encontró el archivo '{pdf_path}' en esta carpeta. Por favor colócalo aquí.")

# Cargador de PDFs oficial de LangChain
loader = PyPDFLoader(pdf_path)
documentos = loader.load()

# Dividimos el contenido extraído del PDF en fragmentos de 500 caracteres
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documentos)
print(f"PDF cargado exitosamente. Se dividió en {len(chunks)} fragmentos.")

# --- PASO 2: Vectorización (FAISS en GPU) ---
print("\n[Paso 2] Indexando fragmentos en FAISS (usando GPU)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cuda'}
)
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("faiss_index_pdf")
print("Base de datos vectorial FAISS creada.")

# --- PASO 3: Recuperación (Retriever) ---
print("\n[Paso 3] Preparando el recuperador y el Prompt...")

# Pregunta
pregunta = "¿Cuál es la hora de ingreso?"

print(f"Pregunta formulada: '{pregunta}'")

# FAISS busca los 3 fragmentos del PDF matemáticamente más cercanos a la pregunta
docs_recuperados = vector_db.similarity_search(pregunta, k=3)
contexto_combinado = "\n\n".join([doc.page_content for doc in docs_recuperados])

# --- PASO 4 y 5: Generación (Gemma 4) y Medición (Baseline) ---
print("\n[Paso 4 y 5] Enviando contexto a Ollama (Gemma 4 E4B) y midiendo rendimiento...")

# Estructura del Prompt con el contexto inyectado del PDF
prompt_sistema = f"""Utiliza estrictamente el siguiente contexto extraído de un documento PDF para responder la pregunta. Si el contexto no contiene la información, di que no lo sabes.

Contexto:
{contexto_combinado}

Pregunta: {pregunta}
Respuesta detallada en español:"""

llm = OllamaLLM(model="gemma4:e4b")

# Tiempo exacto que tarda Gemma 4 en procesar todo
tiempo_inicio = time.time()
respuesta_modelo = llm.invoke(prompt_sistema)
tiempo_total = time.time() - tiempo_inicio

print("\n" + "="*40 + " RESPUESTA DEL MODELO GEMMA 4 " + "="*40)
print(respuesta_modelo)
print("="*110)
print(f"\nBaseline de Tiempo: El proceso de generación con el PDF tardó exactamente {tiempo_total:.2f} segundos.")
print("\n=== SIMULACRO DE PIPELINE COMPLETO CON ÉXITO ===")