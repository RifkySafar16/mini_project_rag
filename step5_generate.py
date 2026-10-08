from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from questions import QUESTIONS

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "aisingapore/Gemma-SEA-LION-v3-9B-IT"
TOP_K = 3

# Grounding instruction sesuai modul Section 5
GROUNDING_PROMPT = """Anda adalah asisten yang menjawab pertanyaan HANYA berdasarkan konteks yang diberikan.
Jika konteks tidak mengandung cukup informasi untuk menjawab, katakan dengan jelas bahwa jawaban tidak dapat ditentukan dari dokumen yang tersedia.
Jangan mengarang jawaban atau menambahkan informasi dari luar konteks.

Konteks:
{context}

Pertanyaan:
{question}
"""


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="faq_perpustakaan",
    )


def retrieve(vs, query, k=TOP_K):
    return vs.similarity_search(query, k=k)


def build_context(docs):
    """Gabungkan chunk jadi satu string konteks, dengan penanda sumber."""
    parts = []
    for i, doc in enumerate(docs, 1):
        section = doc.metadata.get("section_title", "?")
        source = doc.metadata.get("source", "?")
        parts.append(f"[Sumber {i}: {section} — {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def generate_answer(llm, question, docs):
    context = build_context(docs)
    prompt = GROUNDING_PROMPT.format(context=context, question=question)

    messages = [
        SystemMessage(content="Anda adalah asisten yang teliti dan hanya menjawab berdasarkan konteks."),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    return response.content


def main():
    vs = load_vectorstore()
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,      # 0 = deterministik, tidak kreatif
        num_predict=512,    # batas panjang jawaban
    )

    for q in QUESTIONS:
        print("=" * 70)
        print(f"[Q{q['id']}] ({q['type']})")
        print(f"Pertanyaan: {q['question']}")
        print("-" * 70)

        docs = retrieve(vs, q["question"], k=TOP_K)

        print("Konteks yang di-retrieve:")
        for i, doc in enumerate(docs, 1):
            sec = doc.metadata.get("section_title", "?")
            print(f"  #{i} {sec}")
        print()

        answer = generate_answer(llm, q["question"], docs)
        print("Jawaban LLM:")
        print(answer)
        print("=" * 70)
        print()


if __name__ == "__main__":
    main()