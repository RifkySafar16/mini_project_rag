from pathlib import Path
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = Path("data/faq_perpustakaan.txt")
CHROMA_DIR = "chroma_db"


def load_documents():
    docs = TextLoader(str(DATA_PATH), encoding="utf-8").load()
    print(f"[LOAD] {len(docs)} dokumen, {len(docs[0].page_content)} karakter")
    return docs


def split_by_section(docs):
    text = docs[0].page_content
    source = docs[0].metadata["source"]

    sections = re.split(r"(?=^\d+\.\s)", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    for i, section in enumerate(sections):
        m = re.match(r"^(\d+\.\s[^\n]+)", section)
        title = m.group(1) if m else f"Section {i+1}"
        chunks.append(Document(
            page_content=section,
            metadata={"source": source, "section_id": i + 1, "section_title": title},
        ))

    print(f"[SPLIT] {len(chunks)} chunks:")
    for c in chunks:
        print(f"   - {c.metadata['section_title']}")
    return chunks


def create_embeddings():
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print(f"[EMBED] Dimensi: {len(emb.embed_query('test'))}")
    return emb


def build_index(chunks, embeddings):
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="faq_perpustakaan",
    )
    print(f"[INDEX] {len(chunks)} chunks tersimpan di '{CHROMA_DIR}'")
    return vs


def inspect_sample_chunks(chunks, n=3):
    print("\n=== SAMPLE CHUNKS (Checkpoint Step 2) ===")
    for c in chunks[:n]:
        print(f"\n--- {c.metadata['section_title']} ---")
        print(f"Source : {c.metadata['source']}")
        print(f"Section: {c.metadata['section_id']}")
        print(f"Isi    : {c.page_content[:200]}...")
        print(f"Panjang: {len(c.page_content)} karakter")


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_by_section(docs)
    embeddings = create_embeddings()
    vectorstore = build_index(chunks, embeddings)
    inspect_sample_chunks(chunks, n=3)

    # Sanity check: retrieval sederhana
    print("\n=== SANITY CHECK RETRIEVAL ===")
    query = "berapa denda kalau telat mengembalikan buku"
    results = vectorstore.similarity_search_with_score(query, k=3)
    for doc, score in results:
        print(f"\nScore: {score:.4f} | {doc.metadata['section_title']}")
        print(f"  {doc.page_content[:150]}...")