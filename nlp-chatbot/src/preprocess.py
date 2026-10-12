"""
preprocess.py — Text Preprocessing for the NLP Chatbot
=======================================================

PURPOSE:
    Before we can feed text to a machine learning model, we need to clean
    and normalize it. ML models work with numbers, not raw text. Preprocessing
    ensures that the text is consistent so the model can learn patterns better.

WHY EACH STEP IS NEEDED:
    1. Lowercase:    "College" and "college" should be treated as the same word.
    2. Punctuation:  "college?" and "college" should be treated as the same word.
    3. Tokenization: Splitting text into individual words so we can process each one.
    4. Stop words:   Words like "is", "the", "a" appear everywhere and don't help
                     the model distinguish between intents. Removing them lets the
                     model focus on meaningful words like "location", "timing", etc.
    5. Rejoin:       After processing individual words, we combine them back into
                     a single string that TF-IDF can work with.
"""

import re       # Regular expressions — used to remove punctuation
import string   # Provides a list of all punctuation characters

# -------------------------------------------------------------------------
# STOP WORDS LIST
# -------------------------------------------------------------------------
# These are common English words that appear in almost every sentence.
# They don't carry meaning that helps distinguish one intent from another.
#
# For example, in "Where is the college located?":
#   - "is" and "the" appear in many different types of sentences
#   - "where", "college", "located" are the meaningful words
#
# We use a small, hardcoded list instead of importing NLTK's full list.
# This keeps the project simple and avoids an unnecessary dependency.
# -------------------------------------------------------------------------

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now",
    "d", "ll", "m", "o", "re", "ve", "y"
}


def preprocess_text(text):
    """
    Clean and normalize a single text string for the ML model.

    Steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Tokenize (split into words)
        4. Remove stop words
        5. Rejoin into a single string

    Parameters:
        text (str): The raw input text, e.g., "Where is the college located?"

    Returns:
        str: The cleaned text, e.g., "college located"
    """

    # Step 1: Convert to lowercase
    # "College" and "college" should be treated as the same word
    text = text.lower()

    # Step 2: Remove punctuation
    # "college?" becomes "college", "what's" becomes "whats"
    # re.sub replaces all characters that match the pattern with an empty string
    # [^\w\s] means "any character that is NOT a word character or whitespace"
    text = re.sub(r'[^\w\s]', '', text)

    # Step 3: Tokenize — split the text into a list of individual words
    # "where college located" becomes ["where", "college", "located"]
    words = text.split()

    # Step 4: Remove stop words
    # Keep only the words that are NOT in our stop words list
    # This removes common words like "is", "the", "a" that don't help classification
    filtered_words = []
    for word in words:
        if word not in STOP_WORDS:
            filtered_words.append(word)

    # Step 5: Rejoin the remaining words into a single string
    # ["college", "located"] becomes "college located"
    # TF-IDF expects a string as input, not a list of words
    cleaned_text = ' '.join(filtered_words)

    return cleaned_text


# -------------------------------------------------------------------------
# TEST THE PREPROCESSING (only runs when this file is executed directly)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Test with some example sentences to verify preprocessing works correctly
    test_sentences = [
        "Where is the college located?",
        "What are the college hours?",
        "Hello, how are you?",
        "Tell me about the LIBRARY!",
        "Can you tell me what time classes begin?",
    ]

    print("=" * 60)
    print("PREPROCESSING TEST")
    print("=" * 60)

    for sentence in test_sentences:
        cleaned = preprocess_text(sentence)
        print(f"\nOriginal:     {sentence}")
        print(f"Preprocessed: {cleaned}")

    print("\n" + "=" * 60)
