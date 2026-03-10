import streamlit as st
from groq import Groq
from utils import process_pdf, search

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="ChatPDF AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Session State ------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("<h2 style='color:#3B82F6;'>📄 ChatPDF AI</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if uploaded_file:
        if st.session_state.pdf_name != uploaded_file.name:
            try:
                with st.spinner("Processing PDF..."):
                    st.session_state.vectorstore = process_pdf(uploaded_file)
                    st.session_state.pdf_name = uploaded_file.name
                    st.success(f"✅ PDF '{uploaded_file.name}' uploaded successfully!")
            except Exception as e:
                st.error(f"❌ {str(e)}")

    st.markdown("---")
    api_key = st.text_input("Enter Groq API Key", type="password")
    st.markdown("---")

    if st.button("➕ New Chat"):
        chat_id = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[chat_id] = []
        st.session_state.current_chat = chat_id

    st.markdown("### Chats")
    for chat in list(st.session_state.chats.keys()):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(chat, key=f"chat_{chat}"):
                st.session_state.current_chat = chat
        with col2:
            if st.button("❌", key=f"del_{chat}"):
                del st.session_state.chats[chat]
                if st.session_state.current_chat == chat:
                    st.session_state.current_chat = None
                st.experimental_rerun()

# ------------------ Main UI ------------------
st.markdown("<h1 style='color:#1E40AF;'>💬 Chat with your PDF</h1>", unsafe_allow_html=True)

if st.session_state.current_chat:
    chat_history = st.session_state.chats[st.session_state.current_chat]

    # Display chat messages line by line
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            for line in msg["content"].split("\n"):
                st.markdown(line)

    # Chat input
    user_question = st.chat_input("Ask something about the PDF...")
    if user_question:
        if not st.session_state.vectorstore:
            st.error("📄 Upload a PDF first to start chatting.")
        elif not api_key:
            st.error("🔑 Enter your Groq API key to start chatting.")
        else:
            try:
                # Append user message
                chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    for line in user_question.split("\n"):
                        st.markdown(line)

                # Assistant placeholder (no nested chat_message)
                placeholder = st.empty()
                placeholder.markdown("🤖 Thinking...")

               # Retrieve similar chunks
                index, chunks = st.session_state.vectorstore
                context = search(index, chunks, user_question)

                prompt = f"""
                Use the context below to answer the question.
                Context: {context}
                Question: {user_question}
                """

                # Call Groq API
                client = Groq(api_key=api_key)
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You answer questions based on PDF context."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = completion.choices[0].message.content

                # Update placeholder line by line
                placeholder.empty()
                for line in answer.split("\n"):
                    st.chat_message("assistant").markdown(line)

                # Save answer to chat history
                chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
else:
    st.info("📄 Upload a PDF and enter API key to start chatting or start a new chat.")
