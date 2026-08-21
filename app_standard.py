# ============================================================
# DocuIntel STANDARD — ₹7,999 Package
# Features:
# → Digital + Scanned PDF (OCR)
# → Q&A with citations
# → Session memory
# → 2 modes (General + Legal)
# Delivery: 3-4 days | Revisions: 2
# ============================================================

import streamlit as st
import os
import hashlib
import io
from dotenv import load_dotenv
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory


# ============================================================
# SETUP
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)

POPPLER_PATH = (
    r"C:\Release-26.02.0-0\poppler-26.02.0\Library\bin"
)

load_dotenv()

ENV_API_KEY = os.getenv(
    "MISTRAL_API_KEY",
    ""
)

st.set_page_config(
    page_title="DocuIntel Standard",
    page_icon="📋",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_document" not in st.session_state:
    st.session_state.selected_document = None

if "last_selected_document" not in st.session_state:
    st.session_state.last_selected_document = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "db_path" not in st.session_state:
    st.session_state.db_path = None

if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )


# ============================================================
# PROMPTS — 2 modes only
# ============================================================

MODE_PROMPTS = {

    "General": """You are a document assistant.
- Use provided context only
- Bullet points for multiple items
- If not found: 'Not available in document'
- Be concise and clear""",

    "Legal": """You are a legal analyst.
- Use context only, never assume
- Format your answer EXACTLY as:
- Key Finding: 1-2 lines
  Details:
- bullet point
- Risk MUST be exactly one of: Low, Medium, High.
- Never leave Risk blank.
- Choose the risk level based only on the contract context
- If not found: 'Not in document'"""
}


# ============================================================
# FILE SIZE CHECK
# ============================================================

MAX_FILE_SIZE_MB = 10


def check_file_sizes(uploaded_files):

    oversized = []

    for file in uploaded_files:

        size_mb = file.size / (1024 * 1024)

        if size_mb > MAX_FILE_SIZE_MB:

            oversized.append(
                f"{file.name} ({size_mb:.1f} MB)"
            )

    return oversized


# ============================================================
# TEXT EXTRACTION — Digital + Scanned (OCR)
# ============================================================

def extract_text_from_pdfs(uploaded_files):

    all_chunks_meta = []

    for file in uploaded_files:

        file_bytes = file.read()

        try:

            with pdfplumber.open(
                io.BytesIO(file_bytes)
            ) as pdf:

                for page_num, page in enumerate(
                    pdf.pages,
                    start=1
                ):

                    extracted = page.extract_text()

                    # ------------------------------------------------
                    # DIGITAL PDF
                    # ------------------------------------------------
                    if (
                        extracted
                        and len(extracted.strip()) > 50
                    ):

                        all_chunks_meta.append({

                            "text": extracted.strip(),

                            "page": page_num,

                            "source": file.name,

                            "method": "digital"

                        })

                    # ------------------------------------------------
                    # SCANNED PDF → OCR
                    # ------------------------------------------------
                    else:

                        try:

                            page_images = convert_from_bytes(

                                file_bytes,

                                first_page=page_num,

                                last_page=page_num,

                                dpi=300,

                                poppler_path=POPPLER_PATH

                            )

                            if page_images:

                                ocr_text = (
                                    pytesseract.image_to_string(
                                        page_images[0],
                                        lang="eng"
                                    )
                                )

                                if (
                                    ocr_text
                                    and ocr_text.strip()
                                ):

                                    all_chunks_meta.append({

                                        "text": ocr_text.strip(),

                                        "page": page_num,

                                        "source": file.name,

                                        "method": "ocr"

                                    })

                        except Exception:

                            st.warning(
                                f"Page {page_num} of "
                                f"{file.name} skipped."
                            )

                            continue

        except Exception as e:

            st.error(
                f"Error reading {file.name}: {str(e)}"
            )

            continue

        file.seek(0)

    return all_chunks_meta


# ============================================================
# HASHING
# ============================================================

def generate_db_id(uploaded_files):

    file_signatures = []

    for file in uploaded_files:

        file_bytes = file.getvalue()

        file_hash = hashlib.sha256(
            file_bytes
        ).hexdigest()

        file_signatures.append(
            f"{file.name}:{file_hash}"
        )

    hasher = hashlib.sha256()

    for signature in sorted(file_signatures):

        hasher.update(
            signature.encode("utf-8")
        )

    return hasher.hexdigest()[:12]


# ============================================================
# VECTOR DB
# ============================================================

def get_vector_db(
    chunks_meta,
    api_key,
    persist_path
):

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key
    )

    # ------------------------------------------------------------
    # LOAD EXISTING DATABASE
    # ------------------------------------------------------------

    if (
        os.path.exists(persist_path)
        and os.listdir(persist_path)
    ):

        return (

            Chroma(
                persist_directory=persist_path,
                embedding_function=embeddings
            ),

            False

        )

    # ------------------------------------------------------------
    # CREATE NEW DATABASE
    # ------------------------------------------------------------

    texts = []

    metadatas = []

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=150,

        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]

    )

    for item in chunks_meta:

        for chunk in splitter.split_text(
            item["text"]
        ):

            texts.append(chunk)

            metadatas.append({

                "page": item["page"],

                "source": item["source"],

                "method": item.get(
                    "method",
                    "digital"
                )

            })

    vector_db = Chroma.from_texts(

        texts,

        embeddings,

        metadatas=metadatas,

        persist_directory=persist_path

    )

    vector_db.persist()

    return vector_db, True


# ============================================================
# AI RESPONSE — With Memory + Citations
# ============================================================
def get_ai_response(
    query,
    vector_db,
    mode,
    api_key,
    memory,
    selected_document
):
    # ------------------------------------------------------------
    # DOCUMENT MUST BE SELECTED
    # ------------------------------------------------------------
    if not selected_document:
        return (
            "Please select a document before asking a question.",
            []
        )

    # ------------------------------------------------------------
    # SYSTEM PROMPT (Mode-specific)
    # ------------------------------------------------------------
    system_msg = MODE_PROMPTS.get(mode, MODE_PROMPTS["General"])
    
    # Add strict fallback instructions
    system_msg += """

CRITICAL RULES:
1. Answer ONLY using the provided document context.
2. If information is NOT in the context, say EXACTLY:
   "Sorry, I could not find this information in the selected document."
3. Do NOT summarize the document.
4. Do NOT say "I don't know" or any other generic response.
5. Be concise and direct.
"""

    # ------------------------------------------------------------
    # MISTRAL LLM WITH SYSTEM PROMPT
    # ------------------------------------------------------------
    llm = ChatMistralAI(
        model_name="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0
    )

    # ------------------------------------------------------------
    # STRICTLY SEARCH ONLY SELECTED PDF
    # ------------------------------------------------------------
    search_kwargs = {
        "k": 8,
        "filter": {
            "source": selected_document
        }
    }

    retriever = vector_db.as_retriever(
        search_kwargs=search_kwargs
    )

    # ------------------------------------------------------------
    # CONVERSATIONAL RETRIEVAL CHAIN
    # ------------------------------------------------------------
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        verbose=False
    )

    # ------------------------------------------------------------
    # INJECT SYSTEM PROMPT IN USER MESSAGE
    # ------------------------------------------------------------
    formatted_query = f"{system_msg}\n\nUser Question: {query}"
    
    result = chain.invoke({"question": formatted_query})
    answer = result["answer"]

    # ------------------------------------------------------------
    # EXTRACT SOURCES
    # ------------------------------------------------------------
    sources = []
    seen = set()

    for doc in result.get("source_documents", []):
        meta = doc.metadata
        source_name = meta.get("source", "Unknown")
        page_number = meta.get("page", "?")
        method = meta.get("method", "digital")
        tag = "📷 OCR" if method == "ocr" else "📄"
        
        citation = f"{tag} {source_name} — Page {page_number}"
        
        if citation not in seen:
            sources.append(citation)
            seen.add(citation)

    return answer, sources



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Settings")

    # ------------------------------------------------------------
    # API KEY
    # ------------------------------------------------------------

    if ENV_API_KEY:

        st.success(
            "✅ API Key loaded"
        )

        api_key = ENV_API_KEY

    else:

        api_key = st.text_input(
            "Mistral API Key",
            type="password"
        )

    # ------------------------------------------------------------
    # ANALYSIS MODE
    # ------------------------------------------------------------

    mode = st.selectbox(
        "Analysis Mode",
        [
            "General",
            "Legal"
        ]
    )

    st.divider()

    # ------------------------------------------------------------
    # PACKAGE INFO
    # ------------------------------------------------------------

    st.info("""
    📋 **STANDARD Package**
    """)

    st.markdown("""
    ✅ Digital PDF
    ✅ Scanned PDF (OCR)
    ✅ Q&A with citations
    ✅ Session memory
    ✅ 2 modes (General + Legal)
    """)

    # ------------------------------------------------------------
    # CLEAR CHAT
    # ------------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.messages = []

        st.session_state.chat_memory = (
            ConversationBufferMemory(

                memory_key="chat_history",

                return_messages=True,

                output_key="answer"

            )
        )

        st.rerun()


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "📋 DocuIntel Standard"
)

st.caption(
    "AI-powered PDF analyzer — "
    "Digital + Scanned + Memory"
)

c1, c2 = st.columns(2)

c1.success(
    "📄 Digital PDF ✅"
)

c2.success(
    "📷 Scanned PDF ✅"
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(

    "Upload PDFs — Digital & Scanned supported",

    type="pdf",

    accept_multiple_files=True

)


# ============================================================
# DOCUMENT SELECTION
# ============================================================

if uploaded_files:

    document_names = [

        file.name

        for file in uploaded_files

    ]

    selected_document = st.selectbox(

        "📄 Select document for Q&A",

        document_names,

        key="document_selector"

    )

    # --------------------------------------------------------
    # RESET CHAT WHEN SELECTED PDF CHANGES
    # --------------------------------------------------------

    if (
        st.session_state.last_selected_document
        != selected_document
    ):

        st.session_state.messages = []

        st.session_state.chat_memory = (
            ConversationBufferMemory(

                memory_key="chat_history",

                return_messages=True,

                output_key="answer"

            )
        )

        st.session_state.last_selected_document = (
            selected_document
        )

    st.session_state.selected_document = (
        selected_document
    )

else:

    st.session_state.selected_document = None

    st.session_state.last_selected_document = None


# ============================================================
# PROCESSING
# ============================================================

if uploaded_files and api_key:

    oversized = check_file_sizes(
        uploaded_files
    )

    if oversized:

        st.error(
            f"❌ Too large: "
            f"{', '.join(oversized)}"
        )

        st.stop()

    # --------------------------------------------------------
    # DETECT CURRENT PDF SET
    # --------------------------------------------------------

    current_signature = generate_db_id(
        uploaded_files
    )

    # --------------------------------------------------------
    # NEW PDF SET DETECTED
    # --------------------------------------------------------

    if (
        st.session_state.uploaded_signature
        != current_signature
    ):

        st.session_state.uploaded_signature = (
            current_signature
        )

        # Reset vector database
        st.session_state.vector_db = None

        # Reset database path
        st.session_state.db_path = None

        # Reset chat
        st.session_state.messages = []

        # Reset memory
        st.session_state.chat_memory = (
            ConversationBufferMemory(

                memory_key="chat_history",

                return_messages=True,

                output_key="answer"

            )
        )

    # --------------------------------------------------------
    # DATABASE FOR CURRENT PDF SET
    # --------------------------------------------------------

    persist_dir = (
        f"./chroma_standard_"
        f"{current_signature}"
    )

    st.session_state.db_path = (
        persist_dir
    )

    # --------------------------------------------------------
    # CREATE / LOAD VECTOR DATABASE
    # --------------------------------------------------------

    if (
        st.session_state.vector_db
        is None
    ):

        with st.spinner(
            "Processing — Digital + Scanned..."
        ):

            chunks_meta = (
                extract_text_from_pdfs(
                    uploaded_files
                )
            )

            if chunks_meta:

                digital = sum(

                    1

                    for c in chunks_meta

                    if c.get(
                        "method"
                    ) == "digital"

                )

                ocr_count = sum(

                    1

                    for c in chunks_meta

                    if c.get(
                        "method"
                    ) == "ocr"

                )

                vector_db, is_new = (
                    get_vector_db(

                        chunks_meta,

                        api_key,

                        persist_dir

                    )
                )

                st.session_state.vector_db = (
                    vector_db
                )

                if is_new:

                    st.success(

                        f"✅ Done! "
                        f"Digital: {digital} | "
                        f"OCR: {ocr_count} pages"

                    )

                else:

                    st.success(
                        "✅ Loaded existing data!"
                    )

            else:

                st.error(
                    "❌ No text found in PDFs."
                )

else:

    if not api_key:

        st.info(
            "👈 Enter API key."
        )

    elif not uploaded_files:

        st.info(
            "👆 Upload PDFs."
        )


# ============================================================
# CHAT
# ============================================================
if st.session_state.vector_db is not None:
    
    # --------------------------------------------------------
    # SHOW OLD MESSAGES
    # --------------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        st.write(src)
    
    # --------------------------------------------------------
    # USER INPUT (SINGLE SOURCE OF TRUTH)
    # --------------------------------------------------------
    if prompt := st.chat_input("Ask anything about your documents..."):
        
        # --- USER MESSAGE ---
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # --- AI RESPONSE ---
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, sources = get_ai_response(
                    prompt,
                    st.session_state.vector_db,
                    mode,
                    api_key,
                    st.session_state.chat_memory,
                    st.session_state.selected_document
                )
                
                st.markdown(response)
                
                if sources:
                    with st.expander("📚 Sources Used"):
                        for src in sources:
                            st.write(src)
                
                st.download_button(
                    "📥 Download Response",
                    response,
                    file_name="response_standard.txt"
                )
        
        # --- SAVE TO HISTORY ---
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })