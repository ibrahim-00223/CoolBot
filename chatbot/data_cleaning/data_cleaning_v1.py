import os
import glob
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# STEP 1 : 

def doc_loading(files_path="sources"):
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
    documents = []
    for loader in all_files:
        documents.extend(loader.load())
    print(f"{len(documents)} pages chargées. {len(all_files)} PDFs traités.")
    return documents

# STEP 2 : 

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # Taille de chaque chunk (en caractères)
        chunk_overlap=chunk_overlap,      # Chevauchement entre chunks
        length_function=len,    # Comment mesurer la longueur
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]  # Où couper en priorité
)

    chunks = text_splitter.split_documents(documents)
    print(f"Nombre total de chunks : {len(chunks)}.")
    return chunks

# STEP 3 : 
