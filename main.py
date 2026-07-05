import argparse
from pdf_loader import load_pdf
from chunker import split_into_chunks
from embeddings import init_embedding_client, embed_chunks, embed_query
from retriever import retrieve_top_k
from llm_client import init_llm_client, answer_question


def parse_args():
    parser = argparse.ArgumentParser(description="PDF Q&A assistant")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--show-chunks", action="store_true",
                        help="Print retrieved chunks alongside answers")
    return parser.parse_args()


def ingest_pdf(embed_client, pdf_path, chunk_size, overlap):
    full_text = load_pdf(pdf_path)
    chunks = split_into_chunks(full_text, chunk_size=chunk_size, overlap=overlap)

    print("[Info] Generating embeddings...")
    chunk_embeddings = embed_chunks(embed_client, chunks)
    print(f"[Info] Embeddings ready — shape: {chunk_embeddings.shape}\n")

    return chunks, chunk_embeddings


def qa_loop(embed_client, llm_client, chunks, chunk_embeddings, top_k, show_chunks):
    history = []

    print("=" * 50)
    print("  PDF Q&A — type 'exit' to quit")
    print("=" * 50 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        query_emb = embed_query(embed_client, question)

        result = retrieve_top_k(
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            query_embedding=query_emb,
            top_k=top_k,
        )

        if show_chunks:
            print("\n--- Retrieved chunks ---")
            for rank, (chunk, score, idx) in enumerate(
                zip(result.chunks, result.scores, result.indices), 1
            ):
                preview = chunk[:180].replace("\n", " ")
                print(f"  [{rank}] chunk #{idx} (score={score:.4f})")
                print(f"      {preview!r}...\n")

        answer = answer_question(
            client=llm_client,
            question=question,
            context_chunks=result.chunks,
            history=history,
        )

        print(f"\nAssistant: {answer}\n")
        history.append((question, answer))


def main():
    args = parse_args()

    embed_client = init_embedding_client()
    llm_client = init_llm_client()

    chunks, chunk_embeddings = ingest_pdf(
        embed_client, args.pdf_path, args.chunk_size, args.overlap
    )

    qa_loop(embed_client, llm_client, chunks, chunk_embeddings, args.top_k, args.show_chunks)


if __name__ == "__main__":
    main()
