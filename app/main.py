from app.services.llm_client import ask_llm


def main():
    print("Welcome to ResearchPilot!")
    print("Type 'exit' to quit.\n")

    conversation_history = []

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer = ask_llm(question, conversation_history)

        print("\nResearchPilot:", answer, "\n")


if __name__ == "__main__":
    main()