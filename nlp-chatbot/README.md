# NLP-Based Intent Classification and Knowledge Retrieval Chatbot

**3rd Year B.Tech NLP Mini-Project**

A simple, fully understandable chatbot that uses **TF-IDF + Logistic Regression** to classify user questions into intents and retrieve relevant information from a knowledge base.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Flow Diagram](#data-flow-diagram)
3. [Project Structure](#project-structure)
4. [How to Install and Run](#how-to-install-and-run)
5. [Example Inputs and Outputs](#example-inputs-and-outputs)
6. [How Each Component Works](#how-each-component-works)
7. [Explanation of Key Concepts](#explanation-of-key-concepts)
8. [Possible Viva Questions](#possible-viva-questions)
9. [Limitations](#limitations)
10. [Future Improvements](#future-improvements)

---

## Project Overview

This chatbot does **NOT** work by matching predefined questions to predefined answers. Instead, it has two separate components:

1. **Intent Classifier (ML Model)**: Learns language patterns from training examples and predicts what the user is asking about (location? timing? courses?).
2. **Knowledge Base (Dictionary Lookup)**: Contains factual information about the college, organized by topic.

When a user asks a question:
- The ML model figures out the **topic** (intent)
- The knowledge base provides the **answer**

This separation is important — the model learns to understand language patterns, while the knowledge base stores actual information.

---

## Data Flow Diagram

```
User types a question
        |
        v
+------------------+
| Text Preprocessing|    lowercase, remove punctuation,
| (preprocess.py)   |    tokenize, remove stop words
+------------------+
        |
        v
+------------------+
| TF-IDF Vectorizer|    convert text to numerical features
| (trained)        |    (word importance scores)
+------------------+
        |
        v
+------------------+
| Logistic         |    predict the intent + confidence score
| Regression       |
| (trained)        |
+------------------+
        |
        v
+------------------+
| Confidence Check |    Is the model confident enough?
+------------------+
      /          \
     v            v
 YES (>=0.3)   NO (<0.3)
     |            |
     v            v
+----------+  "I'm not sure I
| Knowledge|   understood your
| Base     |   question..."
| Lookup   |
+----------+
     |
     v
  Response
```

---

## Project Structure

```
nlp-chatbot/
|
+-- data/
|   +-- intents.json       # Training data: example sentences + intent labels
|   +-- knowledge.json     # Knowledge base: intent -> factual information
|
+-- src/
|   +-- preprocess.py      # Text preprocessing functions
|   +-- train.py           # Train and save the ML model
|   +-- chatbot.py         # Chatbot logic (predict + retrieve)
|   +-- main.py            # Interactive terminal chatbot
|   +-- test_chatbot.py    # Automated test script
|
+-- models/
|   +-- vectorizer.pkl     # Saved TF-IDF vectorizer (created after training)
|   +-- classifier.pkl     # Saved Logistic Regression model (created after training)
|
+-- requirements.txt       # Python dependencies
+-- README.md              # This file
```

---

## How to Install and Run

### Prerequisites

- Python 3.8 or higher

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

This installs only two packages:
- `scikit-learn` — for TF-IDF and Logistic Regression
- `joblib` — for saving/loading the trained model

### Step 2: Train the model

```bash
cd src
python train.py
```

This will:
1. Load training data from `data/intents.json`
2. Preprocess all sentences
3. Train TF-IDF Vectorizer + Logistic Regression
4. Print a classification report (precision, recall, F1-score)
5. Save the trained model to `models/`

### Step 3: Run the chatbot

```bash
python main.py
```

### Step 4 (Optional): Run automated tests

```bash
python test_chatbot.py
```

---

## Example Inputs and Outputs

### Successful intent recognition (different phrasings, same intent):

```
You: Where is the college?
  [DEBUG] Predicted: 'location' with confidence 0.54
Bot: Vidya Vihar Institute of Technology is located on Science City Road,
     Ahmedabad, Gujarat 380060...

You: Which city is the institute located in?
  [DEBUG] Predicted: 'location' with confidence 0.38
Bot: Vidya Vihar Institute of Technology is located on Science City Road...

You: Can you tell me the college location?
  [DEBUG] Predicted: 'location' with confidence 0.36
Bot: Vidya Vihar Institute of Technology is located on Science City Road...
```

### Different intents:

```
You: What are the library hours?
  [DEBUG] Predicted: 'library' with confidence 0.51
Bot: The college library contains more than 15,000 books...

You: How do I apply for admission?
  [DEBUG] Predicted: 'admission' with confidence 0.63
Bot: Admissions are conducted through the ACPC centralized counseling...
```

### Unknown / out-of-scope query:

```
You: What is the weather today?
  [DEBUG] Predicted: 'facilities' with confidence 0.15 (below threshold 0.3)
Bot: I'm sorry, I'm not sure I understood your question.
     Could you please rephrase it?...
```

---

## How Each Component Works

### 1. `preprocess.py` — Text Preprocessing

Cleans raw text before feeding it to the ML model.

| Step | Operation | Example | Why |
|------|-----------|---------|-----|
| 1 | Lowercase | "College" → "college" | So the model treats "College" and "college" as the same word |
| 2 | Remove punctuation | "college?" → "college" | So "college?" and "college" match |
| 3 | Tokenize | "college located" → ["college", "located"] | Split into individual words for filtering |
| 4 | Remove stop words | ["where", "is", "the", "college"] → ["college"] | Remove common words that don't help classification |
| 5 | Rejoin | ["college", "located"] → "college located" | TF-IDF expects a string, not a list |

### 2. `train.py` — Model Training

Trains two components:

**TF-IDF Vectorizer:**
- Learns the vocabulary (all unique words in training data)
- Computes IDF scores (how rare each word is)
- Transforms sentences into numerical vectors

**Logistic Regression:**
- Learns weights for each word-intent combination
- During prediction, multiplies TF-IDF scores by weights
- Outputs probability for each intent

### 3. `chatbot.py` — Prediction + Retrieval

- `predict_intent()`: Preprocesses input → TF-IDF transform → get probabilities
- `get_response()`: Checks confidence threshold → retrieves from knowledge base

### 4. `main.py` — Interactive Loop

Simple while loop that takes user input and prints bot response.

---

## Explanation of Key Concepts

### What is TF-IDF?

**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**.

It converts text into numbers that represent how important each word is.

**Term Frequency (TF):** How often a word appears in ONE sentence.
- "library library books" → TF of "library" = 2/3 = 0.67

**Inverse Document Frequency (IDF):** How rare a word is across ALL sentences.
- If "library" appears in 10 out of 100 sentences → it's somewhat rare → high IDF
- If "college" appears in 80 out of 100 sentences → it's very common → low IDF

**TF-IDF = TF × IDF**
- A word that is frequent in one sentence but rare overall gets a HIGH score.
- A word that appears everywhere gets a LOW score.

**After TF-IDF, each sentence becomes a vector of numbers:**
```
"Tell me about the library" → [0.0, 0.0, 0.52, 0.0, 0.71, 0.0, ...]
                                        ^              ^
                                      "about"       "library"
```

### Why convert text to numbers?

Machine learning algorithms can only work with numbers, not text. TF-IDF is a way to represent the **meaning** of text using numbers. Words that are more important to a sentence get higher numbers.

### What is Logistic Regression?

Despite its name, Logistic Regression is a **classification** algorithm. It works by:

1. Learning a **weight** for each word for each intent during training
2. For a new sentence, it multiplies each word's TF-IDF score by its weight
3. It applies a **softmax function** to convert the results into probabilities
4. The intent with the **highest probability** is the prediction

Example:
```
Input: "What are the library timings?"

After TF-IDF: [0.0, 0.0, 0.52, 0.0, 0.71, 0.0, ...]

Model computes probabilities:
  greeting   = 0.02
  location   = 0.05
  timing     = 0.08
  library    = 0.51  ← HIGHEST
  courses    = 0.04
  ...

Prediction: "library"
```

### How does the confidence threshold work?

The model always makes a prediction, even for nonsensical input. But the probabilities tell us how confident it is.

- **Known query**: "Where is the college?" → location=0.54 (confident)
- **Unknown query**: "What is the weather?" → facilities=0.15 (not confident)

If the highest probability is below 0.3 (our threshold), we say "I don't understand" instead of giving a wrong answer.

### Why is the knowledge base separate from training data?

- **Training data** teaches the model to recognize language patterns (words like "where", "location", "situated" = location intent)
- **Knowledge base** stores actual facts ("The college is in Ahmedabad")

This separation means:
- You can add new facts without retraining the model
- You can update information without changing the training data
- The model learns **patterns**, not specific answers

---

## Possible Viva Questions

### Q1: What is NLP?
**A:** NLP (Natural Language Processing) is a field of AI that deals with the interaction between computers and human language. It helps computers understand, interpret, and generate human language.

### Q2: What is intent classification?
**A:** Intent classification is the task of determining the purpose or category of a user's message. For example, "Where is the college?" has the intent "location" — the user wants to know the location.

### Q3: Why do we need text preprocessing?
**A:** Raw text contains inconsistencies (uppercase/lowercase, punctuation) and noise (stop words) that make it harder for the model to learn patterns. Preprocessing normalizes the text so the model can focus on meaningful words.

### Q4: What does TF-IDF do?
**A:** TF-IDF converts text into numerical vectors where each number represents how important a word is. Words that are frequent in a specific sentence but rare across all sentences get higher scores. This helps the model identify which words are distinctive for each intent.

### Q5: Why can't we feed raw text directly to the ML model?
**A:** Machine learning algorithms only work with numerical data. We need TF-IDF (or another method) to convert text into a numerical representation that captures the meaning of the text.

### Q6: What is Logistic Regression and why did you choose it?
**A:** Logistic Regression is a simple classification algorithm that learns weights for each feature and produces probability scores. We chose it because: (1) It's simple and interpretable, (2) It works well for text classification with TF-IDF features, (3) It gives probability scores that we can use for confidence checking.

### Q7: How does the model handle questions it hasn't seen before?
**A:** The model generalizes based on word patterns, not exact matches. For example, if it learned from "Where is the college located?", it can also understand "Tell me the location of the institute" because TF-IDF captures that "location", "located", etc. have high scores for the location intent.

### Q8: What happens when the model doesn't understand a question?
**A:** We use the confidence threshold. If the highest probability from the model is below 0.3, the chatbot responds with "I'm not sure I understood your question" instead of giving a potentially wrong answer.

### Q9: Why is the knowledge base separate from the training data?
**A:** The training data teaches the model to recognize intents (language patterns). The knowledge base stores actual information. This separation allows: updating facts without retraining, and keeping the model focused on learning patterns rather than memorizing answers.

### Q10: What are the limitations of this approach?
**A:** (1) Small vocabulary — the model only knows words from training data, (2) No context — each query is independent, no conversation memory, (3) One response per intent — can't handle sub-topics, (4) Requires manual training data, (5) Can't handle misspellings well.

### Q11: What is the difference between fit_transform() and transform()?
**A:** `fit_transform()` learns the vocabulary and IDF scores from the data AND transforms it. `transform()` only transforms new data using the already-learned vocabulary. We use `fit_transform()` on training data and `transform()` on new inputs, because new inputs must be represented using the same vocabulary.

### Q12: What is a stop word?
**A:** A stop word is a common word (like "is", "the", "a", "and") that appears in almost every sentence and doesn't carry meaningful information for distinguishing between intents. Removing them helps TF-IDF focus on the words that actually matter.

### Q13: Why do we split data into training and test sets?
**A:** To evaluate if the model can generalize to new, unseen sentences. If we only test on training data, the model might just be memorizing. Testing on held-out data shows how well it actually learned the patterns.

### Q14: Can this chatbot handle follow-up questions?
**A:** No. Each query is treated independently. The chatbot has no memory of previous questions. Adding conversation context would be a future improvement.

### Q15: How would you improve this chatbot?
**A:** (1) Add more training data, (2) Handle misspellings, (3) Add conversation context/memory, (4) Use sub-intents for more detailed responses, (5) Add a web interface, (6) Use word embeddings for better generalization.

---

## Limitations

1. **Small vocabulary**: The model only knows words present in training data. Completely new words are ignored by TF-IDF.
2. **No conversation context**: Each message is processed independently. The chatbot cannot understand follow-up questions like "Tell me more about that."
3. **One response per intent**: Each intent maps to a single response. The chatbot cannot answer sub-questions within a topic.
4. **No spelling correction**: Misspelled words are treated as unknown words.
5. **Limited training data**: Only ~100 training examples across 8 intents. More data would improve accuracy.
6. **English only**: The model only works with English text.
7. **No entity extraction**: The chatbot cannot extract specific entities (like dates, names) from the query.

---

## Future Improvements

1. **More training data**: Add more diverse example sentences per intent.
2. **Spelling correction**: Use a library like `textblob` or edit-distance algorithms.
3. **Sub-intents**: Split broad intents into more specific ones (e.g., "library_timing" vs "library_books").
4. **Conversation memory**: Track previous messages to handle follow-up questions.
5. **Web interface**: Add a Flask or Streamlit-based web UI.
6. **Entity extraction**: Extract specific information like course names, dates, etc.
7. **Synonym handling**: Use WordNet or similar to expand word matching.
8. **Feedback loop**: Let users indicate if a response was helpful and use that to improve.

---

## Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Programming Language | Python 3 | All code |
| Text Preprocessing | Python built-ins + regex | Clean and normalize text |
| Feature Extraction | scikit-learn TfidfVectorizer | Convert text to numerical features |
| Classification | scikit-learn LogisticRegression | Predict intent from features |
| Model Persistence | joblib | Save and load trained model |
| Data Format | JSON | Human-readable data storage |
