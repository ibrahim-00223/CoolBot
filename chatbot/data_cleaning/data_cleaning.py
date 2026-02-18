from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# Set-up : Get llm's api_key with virtuel environnement 
load_dotenv()


# STEP 1 : Chargement des PDFs
def load_pdfs(files_path="./sources"):
    pdf_files = list(Path(files_path).glob("*.pdf"))
    print(f"🔍 Nombre de PDFs trouvés : {len(pdf_files)}")

    documents_loaded = []
    loaded_count = 0

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents_loaded.extend(loader.load())
            loaded_count += 1
            print(f"✅ PDF ajouté : {pdf_file.name}")
        except Exception as error:
            print(f"❌ Erreur avec {pdf_file.name}: {error}")

    print(f"{len(documents_loaded)} pages chargées. {loaded_count} PDFs traités.")
    return documents_loaded


# STEP 2 : Découpage en chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Nombre total de chunks : {len(chunks)}.")
    return chunks


# STEP 3 : Création de la base vectorielle
def create_vector_store(chunks, api_key, save_path="faiss_index"):
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key
    )

    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local(save_path)
    print("✅ Base vectorielle FAISS créée et sauvegardée.")
    return vector_store


# Orchestration : 
def main():
    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    documents = load_pdfs("./sources")
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks, mistral_api_key)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

if __name__ == "__main__":
    main()