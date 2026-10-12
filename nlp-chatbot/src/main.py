"""
main.py — Interactive Terminal Chatbot
========================================

PURPOSE:
    This is the entry point of the chatbot application.
    It loads the trained model and knowledge base, then starts an
    interactive conversation loop in the terminal.

HOW TO RUN:
    1. First train the model:    python train.py
    2. Then run the chatbot:     python main.py

THE CONVERSATION LOOP:
    1. Print a welcome message
    2. Wait for user input
    3. Pass the input to get_response()
    4. Print the chatbot's response
    5. Repeat until the user types "quit" or "exit"
"""

import os
from chatbot import load_model, load_knowledge_base, get_response


def main():
    """
    Main function that runs the interactive chatbot in the terminal.
    """

    # ---------------------------------------------------------------
    # SETUP: Load the trained model and knowledge base
    # ---------------------------------------------------------------

    # Determine file paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, 'models')
    knowledge_path = os.path.join(project_dir, 'data', 'knowledge.json')

    # Check if the model exists
    if not os.path.exists(os.path.join(model_dir, 'classifier.pkl')):
        print("ERROR: Trained model not found!")
        print("Please run 'python train.py' first to train the model.")
        return

    # Load the model (vectorizer + classifier)
    print("Loading the trained model...")
    vectorizer, classifier = load_model(model_dir)

    # Load the knowledge base
    print("Loading the knowledge base...")
    knowledge_base = load_knowledge_base(knowledge_path)

    # ---------------------------------------------------------------
    # CONVERSATION LOOP
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("  VIDYA VIHAR INSTITUTE OF TECHNOLOGY — HELPDESK CHATBOT")
    print("=" * 60)
    print("\nHello! I'm the college helpdesk chatbot.")
    print("Ask me about the college location, timing, courses,")
    print("admission, library, or facilities.")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("-" * 60)

    while True:
        # Get user input
        print()
        user_input = input("You: ").strip()

        # Check if the user wants to exit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nBot: Thank you for chatting! Goodbye!")
            break

        # Skip empty input
        if not user_input:
            print("Bot: Please type a question. I'm here to help!")
            continue

        # Get the chatbot's response
        response = get_response(user_input, vectorizer, classifier, knowledge_base)

        # Display the response
        print(f"\nBot: {response}")


# Run the chatbot when this script is executed directly
if __name__ == "__main__":
    main()
