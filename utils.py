from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
def process_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if not text.strip():
            raise ValueError("PDF contains no readable text.")

        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {str(e)}")
