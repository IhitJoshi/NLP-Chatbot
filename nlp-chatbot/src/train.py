"""
train.py — Train the Intent Classification Model
===================================================

PURPOSE:
    This script trains a machine learning model that can predict the "intent"
    (purpose/category) of a user's sentence. For example:
        "Where is the college?" → intent = "location"
        "What courses are offered?" → intent = "courses"

THE ML PIPELINE:
    Training sentences → Preprocessing → TF-IDF Vectorization → Logistic Regression

WHAT IS TF-IDF?
    TF-IDF stands for "Term Frequency — Inverse Document Frequency".

    - Term Frequency (TF):
      How often a word appears in a single document (sentence).
      If "college" appears 2 times in a sentence of 10 words, TF = 2/10 = 0.2

    - Inverse Document Frequency (IDF):
      How rare or common a word is across ALL documents (sentences).
      If "college" appears in 80 out of 100 sentences, it's very common → low IDF.
      If "library" appears in only 10 out of 100 sentences, it's rare → high IDF.

    - TF-IDF = TF × IDF
      Words that are frequent in one sentence but rare overall get HIGH scores.
      Words that appear everywhere (like "college") get LOW scores.

    This helps the model focus on the words that actually distinguish one intent
    from another.

    After TF-IDF, each sentence becomes a vector (list) of numbers.
    For example: [0.0, 0.52, 0.0, 0.0, 0.71, 0.0, ...]
    Each number represents the TF-IDF score for one word in the vocabulary.

WHAT IS LOGISTIC REGRESSION?
    Logistic Regression is a classification algorithm. Despite its name, it is
    used for classification, not regression.

    It learns a set of weights for each word in the vocabulary. During prediction,
    it multiplies the TF-IDF scores by these weights and applies a function
    (called the sigmoid/softmax function) to produce a probability for each intent.

    The intent with the highest probability is the predicted intent.

    For example:
        Input: "What are the library timings?"
        Probabilities: greeting=0.02, location=0.05, library=0.85, timing=0.08
        Prediction: "library" (highest probability)
"""

import json
import os
import joblib   # Used to save and load the trained model to/from disk

# Import scikit-learn components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import our preprocessing function
from preprocess import preprocess_text


def load_training_data(filepath):
    """
    Load the training data from the intents.json file.

    The JSON file has this structure:
    {
        "intents": [
            {
                "intent": "greeting",
                "examples": ["Hello", "Hi there", ...]
            },
            ...
        ]
    }

    We need to convert this into two flat lists:
        sentences = ["Hello", "Hi there", "Where is the college?", ...]
        labels    = ["greeting", "greeting", "location", ...]

    Parameters:
        filepath (str): Path to the intents.json file.

    Returns:
        sentences (list): List of all training sentences.
        labels (list):    List of corresponding intent labels.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    sentences = []
    labels = []

    for intent_group in data['intents']:
        intent_name = intent_group['intent']

        for example_sentence in intent_group['examples']:
            sentences.append(example_sentence)
            labels.append(intent_name)

    return sentences, labels


def preprocess_all_sentences(sentences):
    """
    Apply text preprocessing to every sentence in the list.

    Parameters:
        sentences (list): List of raw sentences.

    Returns:
        list: List of preprocessed (cleaned) sentences.
    """
    cleaned_sentences = []

    for sentence in sentences:
        cleaned = preprocess_text(sentence)
        cleaned_sentences.append(cleaned)

    return cleaned_sentences


def train_model(sentences, labels):
    """
    Train the TF-IDF + Logistic Regression pipeline.

    Steps:
        1. Preprocess all training sentences
        2. Split data into training set and test set (for evaluation)
        3. Fit the TF-IDF vectorizer on the training sentences
        4. Transform training sentences into TF-IDF vectors
        5. Train Logistic Regression on the TF-IDF vectors
        6. Evaluate on the test set
        7. Retrain on ALL data for the final model

    Parameters:
        sentences (list): All raw training sentences.
        labels (list):    Corresponding intent labels.

    Returns:
        vectorizer:   The fitted TF-IDF vectorizer (needed to transform new inputs).
        classifier:   The trained Logistic Regression model.
    """

    # Step 1: Preprocess all sentences
    print("Step 1: Preprocessing training sentences...")
    cleaned_sentences = preprocess_all_sentences(sentences)

    # Let's see a few examples of preprocessing
    print("\n  Examples of preprocessing:")
    for i in range(min(3, len(sentences))):
        print(f"    Original:     {sentences[i]}")
        print(f"    Preprocessed: {cleaned_sentences[i]}")
        print()

    # Step 2: Split data into training set (80%) and test set (20%)
    # This lets us check if the model can predict intents it hasn't seen during training
    print("Step 2: Splitting data into training (80%) and test (20%) sets...")
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        cleaned_sentences,
        labels,
        test_size=0.2,          # 20% of data is used for testing
        random_state=42,        # Fixed seed for reproducible results
        stratify=labels         # Ensure each intent is proportionally represented
    )
    print(f"  Training samples: {len(train_sentences)}")
    print(f"  Test samples:     {len(test_sentences)}")

    # Step 3: Fit the TF-IDF Vectorizer on the TRAINING sentences
    # "Fitting" means: the vectorizer learns the vocabulary (all unique words)
    # and computes the IDF scores for each word
    print("\nStep 3: Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_sentences)

    # Show the vocabulary size
    vocabulary = vectorizer.get_feature_names_out()
    print(f"  Vocabulary size: {len(vocabulary)} unique words")
    print(f"  First 10 words in vocabulary: {list(vocabulary[:10])}")
    print(f"  TF-IDF matrix shape: {train_vectors.shape}")
    print(f"    -> {train_vectors.shape[0]} sentences, each represented by {train_vectors.shape[1]} features")

    # Step 4: Transform test sentences using the SAME vectorizer
    # Important: we use transform(), NOT fit_transform()
    # The vectorizer must use the same vocabulary it learned from training data
    test_vectors = vectorizer.transform(test_sentences)

    # Step 5: Train Logistic Regression classifier
    print("\nStep 4: Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        max_iter=1000,          # Maximum training iterations (ensure convergence)
        random_state=42         # Fixed seed for reproducibility
    )
    classifier.fit(train_vectors, train_labels)
    print("  Training complete!")

    # Step 6: Evaluate the model on the test set
    print("\nStep 5: Evaluating on test set...")
    test_predictions = classifier.predict(test_vectors)

    # Classification report shows precision, recall, and F1-score per intent
    print("\n  Classification Report:")
    print("  " + "-" * 55)
    report = classification_report(test_labels, test_predictions, zero_division=0)
    # Indent the report for nicer formatting
    for line in report.split('\n'):
        print(f"  {line}")

    # Step 7: Retrain on ALL data for the final production model
    # During evaluation, we used only 80% of data for training.
    # Now we retrain on 100% so the final model has seen all examples.
    print("\nStep 6: Retraining on ALL data for the final model...")
    all_vectors = vectorizer.fit_transform(cleaned_sentences)
    classifier.fit(all_vectors, labels)
    print("  Final model trained on all data!")

    return vectorizer, classifier


def save_model(vectorizer, classifier, model_dir):
    """
    Save the trained model and vectorizer to disk using joblib.

    We save two files:
        - vectorizer.pkl: The TF-IDF vectorizer (knows the vocabulary and IDF scores)
        - classifier.pkl: The Logistic Regression model (knows the learned weights)

    Both are needed at prediction time:
        1. Vectorizer converts the new sentence into a TF-IDF vector
        2. Classifier predicts the intent from that vector

    Parameters:
        vectorizer:  The fitted TF-IDF vectorizer.
        classifier:  The trained Logistic Regression model.
        model_dir:   Directory path to save the model files.
    """
    # Create the models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    classifier_path = os.path.join(model_dir, 'classifier.pkl')

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)

    print(f"\n  Vectorizer saved to: {vectorizer_path}")
    print(f"  Classifier saved to: {classifier_path}")


# =========================================================================
# MAIN EXECUTION — runs when you execute: python train.py
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NLP CHATBOT — MODEL TRAINING")
    print("=" * 60)

    # Determine file paths relative to this script's location
    # This ensures the script works regardless of which directory you run it from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_filepath = os.path.join(project_dir, 'data', 'intents.json')
    model_dir = os.path.join(project_dir, 'models')

    # Load training data
    print(f"\nLoading training data from: {data_filepath}")
    sentences, labels = load_training_data(data_filepath)
    print(f"  Total training examples: {len(sentences)}")

    # Count examples per intent
    intent_counts = {}
    for label in labels:
        intent_counts[label] = intent_counts.get(label, 0) + 1
    print("  Examples per intent:")
    for intent, count in sorted(intent_counts.items()):
        print(f"    {intent}: {count}")

    # Train the model
    print("\n" + "-" * 60)
    vectorizer, classifier = train_model(sentences, labels)

    # Save the trained model
    print("\n" + "-" * 60)
    print("Saving the trained model...")
    save_model(vectorizer, classifier, model_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
