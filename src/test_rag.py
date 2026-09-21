from search import RAGSearch


def main():

    print("Loading RAG system...")

    rag = RAGSearch()

    print("RAG is ready.")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            answer = rag.search_and_summarize(
                query=question,
                top_k=5
            )

            print("\nAnswer:")
            print(answer)
            print("\n" + "-" * 70 + "\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()