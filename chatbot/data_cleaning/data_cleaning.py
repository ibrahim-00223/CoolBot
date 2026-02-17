from dotenv import load_dotenv
import os
import glob
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
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
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="Chatbot_DataBase" #Name of Collection
)
query = "Je suis sur 30WG et j'aimerai savoir que signifie l'alarme 15001 et quelle est la cause éventuelle ?"
results = vector_store.similarity_search(query, k=3)

retriever = vector_store.as_retriever(
    search_type="similarity",  # ou "mmr"
    search_kwargs={"k": 5}
)

# Retriever : 
docs = retriever.invoke("Je suis sur un 30WG et j'aimerai savoir que signifie l'alarme 15001 ?")

# Avec filtres
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,
        "score_threshold": 0.5  # Seuil minimum
    }
)

print(retriever)
