from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from questions import QUESTIONS

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="faq_perpustakaan",
    )
    return vs


def retrieve(vectorstore, query, k=TOP_K):
    return vectorstore.similarity_search_with_score(query, k=k)


def inspect_retrieval(question_item, results):
    print("=" * 70)
    print(f"[Q{question_item['id']}] ({question_item['type']})")
    print(f"Pertanyaan       : {question_item['question']}")
    print(f"Expected section : {question_item['expected_section']}")
    print("-" * 70)

    hit = False
    for rank, (doc, score) in enumerate(results, start=1):
        section = doc.metadata.get("section_title", "?")
        if section == question_item["expected_section"]:
            hit = True
            marker = "  <-- MATCH"
        else:
            marker = ""
        print(f"  #{rank}  score={score:.4f}  | {section}{marker}")
        print(f"       {doc.page_content[:120].replace(chr(10), ' ')}...")
        print()

    if question_item["expected_section"] is None:
        top_score = results[0][1]
        if top_score > 1.0:
            print("  [OK] Top-1 cukup jauh. Evidence kemungkinan tidak cukup.")
        else:
            print("  [WARN] Ada chunk yang cukup dekat. Cek manual.")
    else:
        if hit:
            print(f"  [OK] Expected section ditemukan di Top-{len(results)}.")
        else:
            print(f"  [FAIL] Expected section TIDAK ditemukan di Top-{len(results)}.")

    print("=" * 70)
    print()
    return hit


def main():
    vs = load_vectorstore()
    summary = []

    for q in QUESTIONS:
        results = retrieve(vs, q["question"], k=TOP_K)
        hit = inspect_retrieval(q, results)
        summary.append({
            "id": q["id"],
            "type": q["type"],
            "hit": hit,
            "expected": q["expected_section"],
        })

    print("\n===== RINGKASAN RETRIEVAL =====")
    for s in summary:
        status = "OK" if (s["hit"] or s["expected"] is None) else "FAIL"
        print(f"  Q{s['id']} [{s['type']:14}] -> {status}")


if __name__ == "__main__":
    main()