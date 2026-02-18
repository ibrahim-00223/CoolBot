import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# Configuration de la page
st.set_page_config(
    page_title="Assistant Technique RAG",
    page_icon="🔧",
    layout="wide"
)

# Chargement des variables d'environnement
load_dotenv()

# Titre de l'application
st.title("🔧 Assistant Technique RAG")
st.markdown("*Votre assistant pour la documentation technique des systèmes de régulation*")

# Sidebar pour les informations
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
    Cet assistant utilise l'IA pour répondre à vos questions techniques
    en se basant sur la documentation disponible.
    
    **Couverture :**
    - Systèmes 30WG, 30WGA, 61WG
    - Systèmes 61CWD, 61CWZ, 61XWHZ
    - Manuels d'installation et de régulation
    """)
    
    st.divider()
    
    st.header("🛠️ Conseils d'utilisation")
    st.markdown("""
    - Posez des questions précises
    - Mentionnez le modèle concerné
    - Citez le numéro d'alarme si applicable
    """)

# Initialisation de la session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    with st.spinner("🔄 Chargement de la base de connaissances..."):
        try:
            # Chargement des embeddings
            Mistral_API_KEY = os.getenv("MISTRAL_API_KEY")
            embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key=Mistral_API_KEY
            )
            
            # Chargement de la base vectorielle
            vector_store = FAISS.load_local(
                "data_cleaning/faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Configuration du retriever
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Création du LLM
            llm = ChatMistralAI(
                model="mistral-large-latest",
                mistral_api_key=Mistral_API_KEY,
                temperature=0.2
            )
            
            # Template du prompt
            template = """Tu es l'assistant technique du technicien. Ton job : l'aider à résoudre vite et bien.

DOCUMENTATION TECHNIQUE :
{context}

QUESTION : {question}

RÉPONDS EN SUIVANT CE FORMAT :

🔍 CE QUE J'AI COMPRIS :
[Reformule le problème en 1 phrase]

✅ SOLUTION RAPIDE :
[Étapes numérotées et concrètes]

⚠️ POINTS D'ATTENTION :
[Précautions ou vérifications importantes]

📖 SOURCES :
[Document(s) et page(s) consultés]

Si l'info n'est pas dans la doc, dis : "❌ Info non trouvée dans la documentation disponible. Je suggère de contacter le support technique."
"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Fonction de formatage des documents
            def format_docs(docs):
                formatted = []
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', 'Inconnu')
                    page = doc.metadata.get('page', 'N/A')
                    formatted.append(f"[Document {i} - {source}, Page: {page}]\n{doc.page_content}")
                return "\n\n---\n\n".join(formatted)
            
            # Création de la chaîne RAG
            st.session_state.rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            
            st.session_state.retriever = retriever
            
            st.success("✅ Assistant prêt !")
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement : {e}")
            st.stop()

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("Posez votre question technique..."):
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche dans la documentation..."):
            try:
                # Afficher les documents pertinents (optionnel)
                with st.expander("📚 Documents consultés"):
                    relevant_docs = st.session_state.retriever.invoke(prompt)
                    for i, doc in enumerate(relevant_docs, 1):
                        source = doc.metadata.get('source', 'Inconnu')
                        page = doc.metadata.get('page', 'N/A')
                        st.markdown(f"**Document {i}** : `{source}` - Page {page}")
                        st.caption(doc.page_content[:200] + "...")
                        st.divider()
                
                # Générer la réponse
                response = st.session_state.rag_chain.invoke(prompt)
                st.markdown(response)
                
                # Ajouter à l'historique
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

# Bouton pour réinitialiser la conversation
if st.session_state.messages:
    if st.sidebar.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()