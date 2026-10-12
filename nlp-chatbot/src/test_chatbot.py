"""
test_chatbot.py — Automated Testing for the Chatbot
=====================================================

PURPOSE:
    This script runs a set of predefined test queries through the chatbot
    and shows the predicted intent, confidence, and response for each.

    This is useful for:
        1. Verifying the chatbot works correctly
        2. Demonstrating the chatbot's behavior during a viva
        3. Showing how different phrasings map to the same intent
        4. Showing how unknown queries are handled
"""

import os
from chatbot import load_model, load_knowledge_base, predict_intent, get_response, CONFIDENCE_THRESHOLD


def run_tests():
    """Run a set of test queries and display results."""

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, 'models')
    knowledge_path = os.path.join(project_dir, 'data', 'knowledge.json')

    # Load model and knowledge base
    vectorizer, classifier = load_model(model_dir)
    knowledge_base = load_knowledge_base(knowledge_path)

    # ---------------------------------------------------------------
    # TEST CASES
    # ---------------------------------------------------------------
    # Each test case is: (query, expected_intent_or_"unknown")
    test_cases = [
        # --- Location queries (different phrasings) ---
        ("Where is the college?", "location"),
        ("Which city is the institute located in?", "location"),
        ("Can you tell me the college location?", "location"),

        # --- Timing queries (different phrasings) ---
        ("When does college start?", "timing"),
        ("What time are classes?", "timing"),
        ("When are classes conducted?", "timing"),

        # --- Course queries ---
        ("What courses are available?", "courses"),
        ("Which engineering programs does the college offer?", "courses"),

        # --- Admission queries ---
        ("How do I apply for admission?", "admission"),
        ("What is the admission procedure?", "admission"),

        # --- Library queries ---
        ("Tell me about the library", "library"),
        ("What are the library hours?", "library"),

        # --- Facilities queries ---
        ("Does the college have a sports ground?", "facilities"),
        ("What labs are available?", "facilities"),

        # --- Greeting and Goodbye ---
        ("Hello!", "greeting"),
        ("Bye, thanks!", "goodbye"),

        # --- Unknown / out-of-scope queries (should be low confidence) ---
        ("What is the weather today?", "unknown"),
        ("Tell me a joke", "unknown"),
        ("What is the capital of France?", "unknown"),
    ]

    # ---------------------------------------------------------------
    # RUN TESTS
    # ---------------------------------------------------------------
    print("=" * 70)
    print("  CHATBOT TEST RESULTS")
    print("=" * 70)

    passed = 0
    failed = 0

    for query, expected in test_cases:
        predicted_intent, confidence = predict_intent(
            query, vectorizer, classifier
        )

        # Determine if the query should be treated as "unknown"
        if confidence < CONFIDENCE_THRESHOLD:
            actual_result = "unknown"
        else:
            actual_result = predicted_intent

        # Check if the result matches expectation
        is_correct = (actual_result == expected)
        status = "PASS" if is_correct else "FAIL"

        if is_correct:
            passed += 1
        else:
            failed += 1

        print(f"\n  Query:      \"{query}\"")
        print(f"  Expected:   {expected}")
        print(f"  Predicted:  {predicted_intent} (confidence: {confidence:.2f})")
        print(f"  Result:     [{status}]")

    # Summary
    total = passed + failed
    print("\n" + "=" * 70)
    print(f"  SUMMARY: {passed}/{total} tests passed, {failed}/{total} tests failed")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
