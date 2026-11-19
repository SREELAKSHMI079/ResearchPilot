from app.services.llm_client import ask_llm


def main():
    print("Welcome to ResearchPilot!")

    question = input("\nAsk ResearchPilot something: ")

    answer = ask_llm(question)

    print("\nGemini's Response:")
    print(answer)


if __name__ == "__main__":
    main()