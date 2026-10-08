from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# GANTI sesuai model yang sudah kamu pull
LLM_MODEL = "aisingapore/Gemma-SEA-LION-v3-9B-IT:latest"   # atau "aisingapore/Gemma-SEA-LION-v3-9B-IT"

TOP_K = 3

GROUNDING_PROMPT = """Anda adalah asisten yang menjawab pertanyaan HANYA berdasarkan konteks yang diberikan.
Jika konteks tidak mengandung cukup informasi untuk menjawab, katakan dengan jelas bahwa jawaban tidak dapat ditentukan dari dokumen yang tersedia.
Jangan mengarang jawaban atau menambahkan informasi dari luar konteks.

Konteks:
{context}

Pertanyaan:
{question}
"""

# Test cases: 1 supported (kontrol) + 3 unsupported
TEST_CASES = [
    {
        "id": "S1",
        "kind": "supported",
        "question": "Berapa denda kalau saya telat mengembalikan buku?",
        "expect_abstain": False,
    },
    {
        "id": "U1",
        "kind": "unsupported",
        "question": "Apakah perpustakaan menyediakan layanan antar buku ke rumah?",
        "expect_abstain": True,
    },
    {
        "id": "U2",
        "kind": "unsupported",
        "question": "Berapa biaya sewa ruang diskusi per jam?",
        "expect_abstain": True,
    },
    {
        "id": "U3",
        "kind": "unsupported",
        "question": "Apakah perpustakaan menyediakan daycare untuk anak?",
        "expect_abstain": True,
    },
]

ABSTAIN_KEYWORDS = [
    "tidak dapat ditentukan",
    "tidak disebutkan",
    "tidak ada informasi",
    "tidak terdapat",
    "tidak tersedia",
    "tidak ditemukan",
    "tidak cukup",
    "belum dijelaskan",
    "tidak memuat",
    "tidak dijelaskan",
]


def load_vectorstore():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=emb,
        collection_name="faq_perpustakaan",
    )


def retrieve(vs, query, k=TOP_K):
    return vs.similarity_search_with_score(query, k=k)


def build_context(docs):
    return "\n\n".join(
        f"[Sumber {i}: {d.metadata.get('section_title','?')}]\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def looks_like_abstain(text):
    t = text.lower()
    return any(kw in t for kw in ABSTAIN_KEYWORDS)


def main():
    vs = load_vectorstore()
    llm = ChatOllama(model=LLM_MODEL, temperature=0, num_predict=512)

    print("=" * 72)
    print("STEP 6 — TEST ABSTAIN")
    print(f"LLM: {LLM_MODEL}")
    print("=" * 72)

    results = []

    for case in TEST_CASES:
        docs_with_score = retrieve(vs, case["question"], k=TOP_K)
        docs = [d for d, _ in docs_with_score]
        top_score = docs_with_score[0][1]

        context = build_context(docs)
        prompt = GROUNDING_PROMPT.format(context=context, question=case["question"])

        answer = llm.invoke([
            SystemMessage(content="Anda hanya menjawab berdasarkan konteks."),
            HumanMessage(content=prompt),
        ]).content

        abstained = looks_like_abstain(answer)

        # Evaluasi
        if case["expect_abstain"]:
            status = "PASS" if abstained else "FAIL"
        else:
            status = "PASS" if not abstained else "FAIL"

        results.append((case["id"], case["kind"], status))

        print(f"\n[{case['id']}] ({case['kind']}) -> {status}")
        print(f"  Pertanyaan : {case['question']}")
        print(f"  Top-1 chunk: {docs[0].metadata.get('section_title')}  (score={top_score:.4f})")
        print(f"  Jawaban    : {answer.strip()}")
        print("-" * 72)

    print("\n===== RINGKASAN =====")
    for cid, kind, status in results:
        print(f"  {cid:3} [{kind:11}] -> {status}")

    total = len(results)
    passed = sum(1 for _, _, s in results if s == "PASS")
    print(f"\nTotal: {passed}/{total} lulus")


if __name__ == "__main__":
    main()