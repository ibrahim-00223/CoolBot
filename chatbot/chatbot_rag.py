from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

load_dotenv()

print("Chargement de la base vectorielle FAISS...")

# STEP 1 : Charger la base vectorielle existante

def load_vector_store(faiss_path="data_cleaning/faiss_index"):
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=mistral_api_key
    )

    vector_store = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Base vectorielle chargée avec succès.")
    return vector_store
# STEP 4 : 

def load_vector_store_and_retriever(faiss_path="data_cleaning/faiss_index", k=5):
    vector_store = load_vector_store(faiss_path)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    print("✅ Retriever configuré avec succès.")
    return retriever

# STEP 3 : Création du LLM

def load_llm(model="mistral-large-latest", temperature=0.2):
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    llm = ChatMistralAI(
        model=model,
        mistral_api_key=mistral_api_key,
        temperature=temperature
    )
    print("✅ LLM chargé avec succès.")
    return llm

# STEP 4 : Création du prompt template
template = """Tu es un assistant technique stagiaire qui aide les techniciens sur le terrain.

Ta mission : aider le technicien à résoudre rapidement ses problèmes en te basant sur la documentation technique disponible.

CONSIGNES :
- Sois direct et pratique, pas de blabla inutile
- Si tu trouves l'info dans la doc, donne la solution étape par étape
- Si plusieurs solutions existent, propose-les toutes
- Indique toujours la source (nom du document et page) pour que le technicien puisse vérifier
- Si tu ne trouves pas l'info, dis-le clairement et suggère où chercher
- Utilise un ton professionnel mais accessible, comme un collègue serviable

DOCUMENTATION DISPONIBLE :
{context}

QUESTION DU TECHNICIEN : {question}

TA RÉPONSE (avec sources) :"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# STEP 5 : Fonction pour formater les documents récupérés
def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Inconnu')
        page = doc.metadata.get('page', 'N/A')
        formatted.append(f"[Document {i} - Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# STEP 6 : Création de la chaîne RAG (Retrieval-Augmented Generation)

def create_rag_chain(prompt, faiss_path="data_cleaning/faiss_index", k=5):
    retriever = load_vector_store_and_retriever(faiss_path, k)
    llm = load_llm()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain créée avec succès.")
    return rag_chain

# STEP 7 : Interface conversationnelle

def run_chat(rag_chain, retriever):
    print("\n" + "="*60)
    print("🤖 ASSISTANT TECHNIQUE RAG - Prêt à répondre !")
    print("="*60)
    print("Tapez 'exit' ou 'quit' pour quitter.\n")
    while True:
        user_question = input("\n❓ Votre question : ")

        if user_question.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Au revoir !")
            break

        if not user_question.strip():
            print("⚠️  Veuillez poser une question.")
            continue

        print("\n🔍 Recherche dans la documentation...")

        try:
            relevant_docs = retriever.invoke(user_question)
            print(f"📚 {len(relevant_docs)} documents pertinents trouvés.\n")

            response = rag_chain.invoke(user_question)

            print("💬 Réponse :")
            print("-" * 60)
            print(response)
            print("-" * 60)

        except Exception as e:
            print(f"❌ Erreur : {e}")

# Orchestration : 

def main():
    retriever = load_vector_store_and_retriever()
    rag_chain = create_rag_chain(prompt)
    run_chat(rag_chain, retriever)

if __name__ == "__main__":
    main()