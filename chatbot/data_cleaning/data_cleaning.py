from dotenv import load_dotenv
import os
import glob
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


load_dotenv()

# STEP 1 : 

files_path="./sources"
    # Get all files in Folder "Sources" : 
all_files = []
pdf_files = list(Path(files_path).glob("*.pdf"))
print(f"🔍 Nombre de PDFs trouvés dans le dossier : {len(pdf_files)}")
# Load pdf documents from the folder :
for pdf_file in pdf_files:
    try:
        loader = PyPDFLoader(str(pdf_file))
        all_files.append(loader)
        print(f"1 pdf ajouté : {pdf_file.name}")
    except Exception as error:
        print(f"Erreur avec {pdf_file.name}: {error}")

    # Retrieve all files that have been uploaded :
    documents_loaded = []
    for loader in all_files:
        documents_loaded.extend(loader.load())
    print(f"{len(documents_loaded)} pages chargées. {len(all_files)} PDFs traités.")

# STEP 2 : 

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Taille de chaque chunk (en caractères)
        chunk_overlap=200,      # Chevauchement entre chunks
        length_function=len,    # Comment mesurer la longueur
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]  # Où couper en priorité
)

chunks = text_splitter.split_documents(documents_loaded)
print(f"Nombre total de chunks : {len(chunks)}.")

# STEP 3 : 

# Retrive Mistral_API_KEY :
Mistral_API_KEY = os.getenv("MISTRAL_API_KEY")
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=Mistral_API_KEY
    )
persist_directory = "./chroma_db"

# Créer la base FAISS
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Sauvegarder localement
vector_store.save_local("faiss_index")
print("Base vectorielle FAISS créée.")

# STEP 4 : Charger et utiliser
# vector_store = FAISS.load_local("faiss_index", embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
