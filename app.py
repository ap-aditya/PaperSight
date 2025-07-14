import streamlit as st
import os
import pinecone
import uuid
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient
from datetime import datetime, timezone
from processor import process_pdf_and_create_chunks
from pinecone import ServerlessSpec
from groq import Groq, RateLimitError
from google.api_core import exceptions


load_dotenv()
MAX_NAMESPACES = 90
GEMINI_MODEL = "gemini-1.5-flash"
GROQ_MODEL = "llama-3.1-8b-instant"

st.set_page_config(
    page_title="PaperSight",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1. Document Processing Options")
    analyze_images_checkbox = st.checkbox(
        "Analyze Images (Requires Gemini)",
        value=False,
        help="This will use the Gemini model to describe images found in the PDF."
    )

    st.subheader("2. API Keys")
    st.caption("Leave blank to use the default keys")
    google_api_key_input = st.text_input("Google API Key", type="password", key="google_key_input")
    groq_api_key_input = st.text_input("Groq API Key", type="password", key="groq_key_input")

    st.markdown("---")
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF to analyze", type="pdf")
    st.caption("Your document is processed locally and securely.")

st.title("PaperSight 📄🤖")
st.subheader("Your AI-powered PDF assistant")
st.info("Upload a document using the sidebar to get started.")

@st.cache_resource(show_spinner="🔌 Setting up connections...")
def init_connections():
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "doc-mentor-index"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, dimension=384, metric="cosine",
            spec=ServerlessSpec(cloud='aws', region=os.environ["PINECONE_ENVIRONMENT"])
        )
    pinecone_index = pc.Index(index_name)

    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri: raise ValueError("MONGODB_URI environment variable not set.")
    mongo_client = MongoClient(mongo_uri)
    mongo_db = mongo_client["documentor_db"]
    namespaces_collection = mongo_db["namespaces"]

    return embedding_model, pinecone_index, namespaces_collection

try:
    embedding_model, pinecone_index, namespaces_collection = init_connections()
except Exception as e:
    st.error(f"Initialization failed. Check API keys and environment variables. Error: {e}")
    st.stop()

def perform_rag(query: str, index, embed_model, provider: str, api_key: str, namespace: str) -> str:
    qa_model_provider = st.session_state.get("qa_model_provider", "Groq")
    if not api_key:
        return f"⚠️ Please provide a {qa_model_provider} API key in the sidebar or set it in your .env file."

    try:
        query_embedding = embed_model.encode(query).tolist()
        results = index.query(vector=query_embedding, top_k=5, namespace=namespace, include_metadata=True)
        context = "\n---\n".join([res['metadata'].get('text', '') for res in results['matches']])

        prompt = f"""You are an expert Q&A assistant. Based ONLY on the following context, provide a detailed answer.
        If the context does not contain the answer, state that clearly.
        CONTEXT: {context}
        QUESTION: {query}
        ANSWER:"""

        if provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            return response.text
        else: 
            client = Groq(api_key=api_key)
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
            )
            return chat_completion.choices[0].message.content

    except exceptions.ResourceExhausted:
        return "🚦 **Gemini API Quota Exceeded** 🚦\nThe default API key has hit its limit. Please provide your own key in the sidebar or try again later."
    except RateLimitError:
        return "🚦 **Groq API Rate Limit Reached** 🚦\nToo many requests too quickly. Please wait a moment and try again, or provide your own key."
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred. Please check the application logs."


def cleanup_old_namespaces(pinecone_index, namespaces_collection, max_namespaces=MAX_NAMESPACES):
    try:
        count = namespaces_collection.count_documents({})
        if count >= max_namespaces:
            num_to_delete = count - max_namespaces + 1
            oldest_namespaces = namespaces_collection.find().sort("created_at", 1).limit(num_to_delete)
            for ns_doc in oldest_namespaces:
                pinecone_index.delete(delete_all=True, namespace=ns_doc["namespace"])
                namespaces_collection.delete_one({"namespace": ns_doc["namespace"]})
    except Exception as e:
        st.warning(f"Could not perform namespace cleanup: {e}")


if uploaded_file:
    file_id = uploaded_file.file_id

    if "processed_file_id" not in st.session_state or st.session_state.processed_file_id != file_id:
        st.session_state.pinecone_namespace = str(uuid.uuid4())
        cleanup_old_namespaces(pinecone_index, namespaces_collection)

        st.info(f"Analyzing '{uploaded_file.name}'...")
        progress_bar = st.progress(0, text="Starting processing...")

        try:
            google_api_key_for_processing = google_api_key_input
            should_process_images = analyze_images_checkbox and bool(google_api_key_for_processing)
            if analyze_images_checkbox and not should_process_images:
                st.warning("⚠️ Images are being skipped because no Google API Key was provided in the sidebar.")
            # ------------------------------------

            file_bytes = uploaded_file.getvalue()
            chunks_with_meta = process_pdf_and_create_chunks(
                file_bytes, google_api_key_for_processing, should_process_images, progress_bar
            )
            
            if chunks_with_meta:
                progress_bar.progress(100, text="Creating embeddings...")
                chunk_texts = [item['text'] for item in chunks_with_meta]
                embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True).tolist()

                progress_bar.progress(100, text="Upserting to database...")
                ids = [f"chunk_{i}" for i in range(len(chunk_texts))]
                vectors = [{"id": id_, "values": emb, "metadata": {"text": txt}} for id_, emb, txt in zip(ids, embeddings, chunk_texts)]

                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    pinecone_index.upsert(vectors=vectors[i:i + batch_size], namespace=st.session_state.pinecone_namespace)

                namespaces_collection.insert_one({
                    "namespace": st.session_state.pinecone_namespace,
                    "created_at": datetime.now(timezone.utc),
                    "document_name": uploaded_file.name
                })

                st.session_state.processed_file_id = file_id
                st.session_state.processed_file_name = uploaded_file.name
                progress_bar.empty()
                st.success(f"✅ '{uploaded_file.name}' processed successfully!")
            else:
                progress_bar.empty()
                st.error("No content could be extracted from the document.")
        except Exception as e:
            progress_bar.empty()
            st.error(f"Processing error: {e}")

if "processed_file_name" in st.session_state:
    st.info(f"Document '{st.session_state.processed_file_name}' is ready for questions.")
    
    qa_model_provider = st.selectbox(
        "Choose a model for Q&A",
        ("Groq", "Gemini"),
        index=0,
        help="You can switch models at any time for asking questions.",
        key="qa_model_provider"
    )

    if st.text_input(f"Ask a question using {qa_model_provider}:", key="qa_input"):
        question = st.session_state.qa_input
        with st.chat_message("user"):
            st.write(question)

        with st.spinner(f"🔍 Searching for answers with {qa_model_provider}..."):
            if qa_model_provider == "Gemini":
                api_key_to_use = google_api_key_input or os.getenv("GOOGLE_API_KEY")
            else:
                api_key_to_use = groq_api_key_input or os.getenv("GROQ_API_KEY")

            answer = perform_rag(question, pinecone_index, embedding_model, qa_model_provider, api_key_to_use, st.session_state.pinecone_namespace)
            with st.chat_message("assistant"):
                st.write(answer)

st.markdown("---")
st.caption("PaperSight © 2025 | Built with Streamlit")