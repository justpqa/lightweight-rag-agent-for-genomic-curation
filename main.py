from ingest import ingest_corpus
from rag import RAG 

if __name__ == "__main__":
    try:
        while True:
            print("Welcome to the Genomic Curation bot")
            print("""Do you want to re-ingesting the documents database or start a chat\n[1] Ingesting corpus\n[2] Start a chat\n[3] Quit""")
            user_choice = ""
            while user_choice == "":
                user_input = input("> ")
                user_input = user_input.strip().lower()
                if user_input == "1":
                    user_choice = "1"
                elif user_input == "2":
                    user_choice = "2"
                elif user_input == "3":
                    user_choice = "3"
                else:
                    print("You should reply by typing 1 or 2 or 3 (or Ctrl-C to exit)")
            if user_choice == "1":
                ingest_corpus()
            elif user_choice == "2":
                rag = RAG()
                print("Starting a new chat, type your question, or type 'quit' if you want to exit")
                while True:
                    print(">A: Hello, what do you want to ask me today (or type 'quit' if you want to exit to main menu)")
                    user_input = input(">Q: ")
                    if user_input.strip() == "quit":
                        break
                    else:
                        answer = rag.answer(user_input.strip())
                        print(f""">A: {answer}""")
                        print()
            elif user_choice == "3":
                print("\nExiting. Goodbye!")
                break
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")