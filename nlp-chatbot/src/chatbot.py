"""
chatbot.py — Chatbot Logic (Prediction + Knowledge Retrieval)
==============================================================

PURPOSE:
    This module connects the trained ML model with the knowledge base.

    When a user asks a question, this module:
        1. Preprocesses the user's input (same cleaning as training data)
        2. Converts it into a TF-IDF vector (using the saved vectorizer)
        3. Predicts the intent using the saved classifier
        4. Checks the confidence of the prediction
        5. Retrieves the relevant response from the knowledge base

DATA FLOW:
    User Input
        ↓
    preprocess_text()         — clean the text
        ↓
    vectorizer.transform()    — convert to TF-IDF numbers
        ↓
    classifier.predict_proba() — get probability for each intent
        ↓
    Check confidence threshold
        ↓
    If confident: look up knowledge base → return response
    If not confident: return "I don't understand" message

CONFIDENCE THRESHOLD:
    The classifier outputs a probability for each intent.
    For example:
        greeting=0.05, location=0.80, timing=0.10, library=0.05

    The highest probability is 0.80 for "location", so the prediction is "location".

    But if the user asks something unrelated like "What is the weather?",
    the probabilities might be:
        greeting=0.15, location=0.18, timing=0.20, library=0.12, ...

    The highest is only 0.20 — the model is not confident about any intent.
    In this case, we should NOT pretend to understand the question.

    We set a CONFIDENCE_THRESHOLD (e.g., 0.4). If the highest probability
    is below this threshold, we respond with "I'm not sure I understood."
"""

import json
import os
import joblib

from preprocess import preprocess_text


# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

# If the model's highest confidence is below this threshold,
# we treat the query as "unknown" and don't guess an intent.
#
# With a small dataset (~100 examples, 8 intents), confidence scores
# are typically between 0.3 and 0.7 for correct predictions.
# Unknown queries usually score below 0.2.
# A threshold of 0.3 balances between catching unknowns and not
# rejecting valid queries.
CONFIDENCE_THRESHOLD = 0.3

# Response for when the model is not confident enough
FALLBACK_RESPONSE = (
    "I'm sorry, I'm not sure I understood your question. "
    "Could you please rephrase it? "
    "I can help with topics like college location, timing, courses, "
    "admission, library, and facilities."
)


def load_model(model_dir):
    """
    Load the trained TF-IDF vectorizer and Logistic Regression classifier
    from disk.

    Parameters:
        model_dir (str): Path to the directory containing the saved model files.

    Returns:
        vectorizer:  The fitted TF-IDF vectorizer.
        classifier:  The trained Logistic Regression model.
    """
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    classifier_path = os.path.join(model_dir, 'classifier.pkl')

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)

    return vectorizer, classifier


def load_knowledge_base(filepath):
    """
    Load the knowledge base from the JSON file.

    The knowledge base maps each intent to a factual response.
    For example:
        {
            "location": "Vidya Vihar Institute is located in Ahmedabad...",
            "timing":   "Classes are conducted from 9 AM to 4 PM...",
            ...
        }

    Parameters:
        filepath (str): Path to the knowledge.json file.

    Returns:
        dict: A dictionary mapping intent names to response strings.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        knowledge = json.load(file)

    return knowledge


def predict_intent(user_input, vectorizer, classifier):
    """
    Predict the intent of a user's input sentence.

    Steps:
        1. Preprocess the input text (same cleaning used during training)
        2. Convert to TF-IDF vector using the trained vectorizer
        3. Get probability scores for each intent
        4. Find the intent with the highest probability

    Parameters:
        user_input (str):  The raw text entered by the user.
        vectorizer:        The fitted TF-IDF vectorizer.
        classifier:        The trained Logistic Regression model.

    Returns:
        predicted_intent (str):  The name of the predicted intent.
        confidence (float):      The probability (0.0 to 1.0) of the prediction.
    """

    # Step 1: Preprocess the input (same as training data)
    cleaned_input = preprocess_text(user_input)

    # Step 2: Convert to TF-IDF vector
    # Note: we use transform(), NOT fit_transform()
    # The vectorizer must use the same vocabulary it learned during training
    # The input must be passed as a list (even for a single sentence)
    input_vector = vectorizer.transform([cleaned_input])

    # Step 3: Get probability scores for ALL intents
    # predict_proba returns an array of probabilities, one per intent
    # For example: [[0.05, 0.02, 0.80, 0.03, 0.05, 0.02, 0.02, 0.01]]
    probabilities = classifier.predict_proba(input_vector)

    # The probabilities array has shape (1, num_intents) for our single input
    # We take the first (and only) row
    intent_probabilities = probabilities[0]

    # Step 4: Find the intent with the highest probability
    # classifier.classes_ contains the intent names in the same order as probabilities
    max_probability_index = intent_probabilities.argmax()
    predicted_intent = classifier.classes_[max_probability_index]
    confidence = intent_probabilities[max_probability_index]

    return predicted_intent, confidence


def get_response(user_input, vectorizer, classifier, knowledge_base):
    """
    Get the chatbot's response for a user's input.

    This is the main function that ties everything together:
        1. Predict the intent
        2. Check if the model is confident enough
        3. Retrieve the answer from the knowledge base

    Parameters:
        user_input (str):      The raw text entered by the user.
        vectorizer:            The fitted TF-IDF vectorizer.
        classifier:            The trained Logistic Regression model.
        knowledge_base (dict): The knowledge base mapping intents to responses.

    Returns:
        response (str): The chatbot's response to the user.
    """

    # Predict the intent and get the confidence score
    predicted_intent, confidence = predict_intent(
        user_input, vectorizer, classifier
    )

    # Check if the model is confident enough
    if confidence < CONFIDENCE_THRESHOLD:
        # The model is not sure — don't guess
        print(f"  [DEBUG] Predicted: '{predicted_intent}' with confidence {confidence:.2f} "
              f"(below threshold {CONFIDENCE_THRESHOLD})")
        return FALLBACK_RESPONSE

    # The model is confident — retrieve the answer from the knowledge base
    print(f"  [DEBUG] Predicted: '{predicted_intent}' with confidence {confidence:.2f}")

    # Look up the predicted intent in the knowledge base
    if predicted_intent in knowledge_base:
        response = knowledge_base[predicted_intent]
    else:
        # This shouldn't normally happen, but it's good to handle it
        response = FALLBACK_RESPONSE

    return response
