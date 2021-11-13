## for data
from datetime import datetime

import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn import svm, metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
## for machine learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler

# download('stopwords')

# WICHTIG: download im terminal mit:
# python -m spacy download de_core_news_lg
nlp = spacy.load("de_core_news_lg")

## for explainer

# This should be our data
# categories = ["OTHER", "INSULT", "ABUSE", "PROFANITY"]
categories = ["INSULT", "ABUSE", "PROFANITY"]

"""
We performed minimal pre-processing by:
• replacing all mentions/usernames with the
generic form User;
• removing the line break characters |LBR|;
• removing the hash character from all hashtags;
(• remove emojis)
• removing stop words using the Python module
stop-words

TODO:
SVM weight!
-> SVM weighting
-> n-gramme
-> features like cites
-> features like #
"""

training_text = []
training_label = []
test_text = []
test_label = []


def main():
    # Train the model
    clf = train()
    # Test the model
    test(clf)


def get_pos(tweet):
    text = ""
    doc = nlp(tweet)
    for token in doc:
        text += str(token.text) + "_" + str(token.tag_) + " "
    return text

def get_data():
    print("Obtain training data...")
    with open("train/germeval2018.training.txt", encoding='utf-8') as f:
    #with open("train/train_.txt", encoding='utf-8') as f:
        file_lines = f.readlines()
        # Extract text and insult classification for each line
        for line in file_lines:
            splitted_line = line.split("\t")
            text = splitted_line[0]
            classification = splitted_line[2].rstrip()
            if classification != "OTHER":
                training_text.append(text)
                training_label.append(classification)
        f.close()

    print("Obtain testing data...")
    with open("train/germeval2018.test_.txt", encoding="utf-8") as f:
    #with open("train/test_.txt", encoding='utf-8') as f:
        file_lines = f.readlines()
        # Extract text and insult classification for each line
        for line in file_lines:
            splitted_line = line.split("\t")
            text = splitted_line[0]
            classification = splitted_line[2].rstrip()
            if classification != "OTHER":
                test_text.append(text)
                test_label.append(classification)
        f.close()


def get_feature_data(text):
    # Clean text data (lemma)
    training_text_clean = []

    # Count of user mentions
    mentions = []

    # Count of hashtags
    hashtags = []

    # Add features
    for item in text:
        # spaCy lemma
        # später pipeline durch aufrufe ändern -> performance
        doc = nlp(item)
        clean = []
        hashtag_count = 0
        mention_count = 0
        for token in doc:
            t = token.text
            lemma = token.lemma_
            # Preprocess text
            if t.__contains__("@"):
                lemma = "USER"
                mention_count += 1
            # Remove hashtags
            elif t.startswith("#"):
                lemma = t[1:]
                hashtag_count += 1

            # Add text if it is not LBR
            if t != "|LBR|":
                clean.append(lemma)

        hashtags.append(hashtag_count)
        mentions.append(mention_count)
        training_text_clean.append(" ".join(clean))

    data_frame = pd.DataFrame(
        {
            "tweets": training_text_clean,
            "hashtags": hashtags,
            "mentions": mentions
        }
    )
    return data_frame


def train():
    get_data()
    print("Get training features...")
    data_frame = get_feature_data(training_text)

    preprocessor = ColumnTransformer(
        [('tweets', TfidfVectorizer(stop_words=stopwords.words('german'), ngram_range=(1, 3)), 'tweets'),
         ('scaler', MinMaxScaler(), ["hashtags", "mentions"])
         ],
        remainder='passthrough', verbose_feature_names_out=True, n_jobs=-1)

    text_clf_svm = Pipeline([
        ('preprocessor', preprocessor),
        ('clf-svm', svm.SVC(kernel='rbf', C=10.0, gamma=0.1, class_weight="balanced"))
    ])

    print("Make model fit...")
    start = datetime.now()
    text_clf_svm = text_clf_svm.fit(data_frame, training_label)
    end = datetime.now()
    duration = end-start
    print("Fit duration: " + str(duration))
    return text_clf_svm


def test(clf):
    print("Get testing features...")
    data_frame = get_feature_data(test_text)
    print("Predict...")

    predicted_svm = clf.predict(data_frame)

    print("Support Vector Maschine:\n" + str(np.mean(predicted_svm == test_label)))
    print(metrics.classification_report(test_label, predicted_svm, target_names=categories))


if __name__ == '__main__':
    main()
