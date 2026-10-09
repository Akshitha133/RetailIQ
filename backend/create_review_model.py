import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "reviews"
)

MODEL_DIR = (
    BASE_DIR
    / "backend"
    / "models"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. DATASET LOCATION
# ==========================================

DATA_FILE = (
    DATA_DIR
    / "reviews.tsv"
)


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("Loading customer review dataset...")


if not DATA_FILE.exists():

    print("\nDataset not found.")

    print(
        "\nPlease download the UCI "
        "Sentiment Labelled Sentences dataset."
    )

    print(
        "\nDownload it from:"
    )

    print(
        "https://archive.ics.uci.edu/dataset/331/"
        "sentiment+labelled+sentences"
    )

    print(
        "\nAfter downloading the ZIP file:"
    )

    print(
        "1. Extract it."
    )

    print(
        "2. Open the 'sentiment labelled sentences' folder."
    )

    print(
        "3. Open the 'amazon' folder."
    )

    print(
        "4. Copy amazon_cells_labelled.txt"
    )

    print(
        "5. Rename it to reviews.tsv"
    )

    print(
        "6. Put it here:"
    )

    print(DATA_FILE)

    raise FileNotFoundError(
        "\nReview dataset is missing."
    )


# UCI format:
# sentence<TAB>label

df = pd.read_csv(
    DATA_FILE,
    sep="\t",
    header=None,
    names=[
        "review",
        "sentiment"
    ]
)


print(
    "Dataset shape:",
    df.shape
)


# ==========================================
# 4. BASIC CLEANING
# ==========================================

print("\nCleaning review text...")


df["review"] = (
    df["review"]
    .astype(str)
    .str.strip()
)


df = df.dropna(
    subset=[
        "review",
        "sentiment"
    ]
)


df = df[
    df["review"].str.len() > 0
]


# ==========================================
# 5. MAP SENTIMENT LABELS
# ==========================================

df["sentiment"] = df[
    "sentiment"
].astype(int)


df["sentiment_label"] = df[
    "sentiment"
].map({
        0: "Negative",
        1: "Positive"
    })


print("\nSentiment distribution:")

print(
    df["sentiment_label"]
    .value_counts()
)


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

print("\nCreating train/test split...")


X = df["review"]

y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    "Training reviews:",
    len(X_train)
)

print(
    "Testing reviews:",
    len(X_test)
)


# ==========================================
# 7. TF-IDF
# ==========================================

print("\nCreating TF-IDF features...")


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    "TF-IDF training shape:",
    X_train_tfidf.shape
)

print(
    "TF-IDF testing shape:",
    X_test_tfidf.shape
)


# ==========================================
# 8. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression model...")


model = LogisticRegression(
    max_iter=1000,
    C=2.0,
    random_state=42
)


model.fit(
    X_train_tfidf,
    y_train
)


print(
    "Model training complete!"
)


# ==========================================
# 9. PREDICTIONS
# ==========================================

print("\nGenerating predictions...")


predictions = model.predict(
    X_test_tfidf
)


# ==========================================
# 10. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n===================================")
print("CUSTOMER SENTIMENT MODEL RESULTS")
print("===================================")


print(
    f"Accuracy: {accuracy:.4f}"
)


print("\nClassification Report:")


print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Negative",
            "Positive"
        ]
    )
)


print("\nConfusion Matrix:")


print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

print("\nFinding influential words...")


feature_names = (
    vectorizer
    .get_feature_names_out()
)


coefficients = (
    model
    .coef_[0]
)


word_importance = pd.DataFrame({
    "word": feature_names,
    "coefficient": coefficients
})


# Strong positive words

positive_words = (
    word_importance
    .sort_values(
        "coefficient",
        ascending=False
    )
    .head(20)
)


# Strong negative words

negative_words = (
    word_importance
    .sort_values(
        "coefficient",
        ascending=True
    )
    .head(20)
)


print("\nTop positive words:")

print(
    positive_words.to_string(
        index=False
    )
)


print("\nTop negative words:")

print(
    negative_words.to_string(
        index=False
    )
)


# ==========================================
# 12. SAVE WORD IMPORTANCE
# ==========================================

importance_file = (
    MODEL_DIR
    / "review_word_importance.csv"
)


word_importance.to_csv(
    importance_file,
    index=False
)


# ==========================================
# 13. SAVE MODEL + VECTORIZER
# ==========================================

model_file = (
    MODEL_DIR
    / "review_sentiment_model.pkl"
)


vectorizer_file = (
    MODEL_DIR
    / "review_tfidf_vectorizer.pkl"
)


joblib.dump(
    model,
    model_file
)


joblib.dump(
    vectorizer,
    vectorizer_file
)


# ==========================================
# 14. TEST SAMPLE PREDICTIONS
# ==========================================

print("\n===================================")
print("SAMPLE REVIEW PREDICTIONS")
print("===================================")


sample_reviews = [
    "The product quality is excellent and delivery was fast.",
    "Very poor quality and terrible customer service.",
    "The product is okay but delivery was slow."
]


sample_features = vectorizer.transform(
    sample_reviews
)


sample_predictions = model.predict(
    sample_features
)


sample_probabilities = model.predict_proba(
    sample_features
)


for review, prediction, probability in zip(
    sample_reviews,
    sample_predictions,
    sample_probabilities
):

    label = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    confidence = probability[
        prediction
    ]

    print("\nReview:")
    print(review)

    print(
        "Prediction:",
        label
    )

    print(
        f"Confidence: {confidence:.2%}"
    )


# ==========================================
# 15. SAVE CLEAN DATASET
# ==========================================

clean_dataset_file = (
    DATA_DIR
    / "clean_reviews.csv"
)


df[
    [
        "review",
        "sentiment_label"
    ]
].to_csv(
    clean_dataset_file,
    index=False
)


# ==========================================
# 16. FINAL OUTPUT
# ==========================================

print("\n===================================")
print("CUSTOMER REVIEW MODEL COMPLETE!")
print("===================================")


print("\nModel saved to:")

print(model_file)


print("\nTF-IDF vectorizer saved to:")

print(vectorizer_file)


print("\nWord importance saved to:")

print(importance_file)


print("\nClean dataset saved to:")

print(clean_dataset_file)