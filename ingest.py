"""
ingest.py — Load the GER-3620-P.pdf, split into chunks,
embed with a local sentence-transformers model,
and store in a local ChromaDB collection.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH = "GER-3620-P.pdf"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "ger_3620"

def ingest():
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"\n❌ PDF not found at '{PDF_PATH}'.\n"
            "   Please place 'GER-3620-P.pdf' in this directory and run again."
        )

    print(f"📄 Loading PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"   → Loaded {len(pages)} pages")

    # Add page-number metadata so we can cite sources later
    for i, page in enumerate(pages):
        page.metadata["page"] = i + 1  # 1-indexed

    print("✂️  Splitting into chunks …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " "],
    )
    docs = splitter.split_documents(pages)
    print(f"   → {len(docs)} chunks created")

    print("🤖 Loading embedding model (all-MiniLM-L6-v2) …")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"💾 Writing to ChromaDB at ./{CHROMA_DIR} …")
    # Always rebuild from scratch so the collection stays fresh
    if Path(CHROMA_DIR).exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"✅ Ingestion complete — {vectorstore._collection.count()} vectors stored.")
    return vectorstore


if __name__ == "__main__":
    ingest()
