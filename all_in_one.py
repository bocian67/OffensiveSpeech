## for data
from datetime import datetime

import numpy as np
import pandas as pd
import regex
import spacy
from matplotlib.colors import Normalize
from nltk.corpus import stopwords
from nltk.metrics import scores
from sklearn import svm, metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
## for machine learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
import emojis
from preprocess_constants import *
import matplotlib.pyplot as plt
from get_dataset import get_splitted_data

# download('stopwords')

# WICHTIG: download im terminal mit:
# python -m spacy download de_core_news_lg
nlp = spacy.load("de_core_news_lg")

## Constants
# This should be our data
categories = ["INSULT", "ABUSE", "PROFANITY", "OTHER"]

# Scores
punctuation_score_dict = {
    "!": 1,
    "?": 0.7,
    "default_emoji": 0.1
}

"""
TODO:
-> SVM weighting
"""

training_text = []
training_label = []
test_text = []
test_label = []
insult_list = []


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def main():
    global insult_list
    insult_list = preprocess_insults()
    emoji_scores = preprocess_emojis()
    punctuation_score_dict.update(emoji_scores)
    # Train the model
    clf = train()
    # Test the model
    test(clf)


# feature_list: ["!", "?", "!", ...]
def scale_feature(feature_list):
    score = 0.0
    for position in range(len(feature_list)):
        if feature_list[position] in punctuation_score_dict:
            emoji_score = punctuation_score_dict[feature_list[position]]
        else:
            emoji_score = punctuation_score_dict["default_emoji"]
        score += (0.5 ** position) * emoji_score
    return score


def get_pos(tweet):
    text = ""
    doc = nlp(tweet)
    for token in doc:
        text += str(token.text) + "_" + str(token.tag_) + " "
    return text


def get_feature_data(text):
    # Clean text data (lemma)
    training_text_clean = []

    # Count of user mentions
    mentions = []

    # Count of hashtags
    hashtags = []

    # Currently not used
    question_mark_numbers = []
    exclamation_mark_numbers = []

    punctuation_scores = []

    emoji_scores = []

    insult_count = []

    # Add features
    for item in text:
        # spaCy lemma
        # später pipeline durch aufrufe ändern -> performance
        doc = nlp(item)
        clean = []
        hashtag_count = 0
        mention_count = 0
        exclamation_mark_score = 0
        question_mark_score = 0
        punctuation = []
        tweet_emojis = []
        insult_item_count = 0
        scaled_emoji_feature = 0

        # Emoji classification
        # Classification Score was not improved; try to use categorical attributes? Maybe with threshold?
        # or use transformer to alter the scores that the SVC use it properly to classify
        emoji_iter = emojis.iter(item)
        for emoji in emoji_iter:
            tweet_emojis.append(emoji)
        if len(tweet_emojis) > 0:
            scaled_emoji_feature = scale_feature(tweet_emojis)
        emoji_scores.append(scaled_emoji_feature)

        for token in doc:
            t = token.text
            lemma = token.lemma_
            # Preprocess text
            # Is text a user mention?
            if t.__contains__("@"):
                lemma = "USER"
                mention_count += 1
            # Is text a hashtag
            elif t.startswith("#"):
                lemma = t[1:]
                t = t[1:]
                hashtag_count += 1
            # Is text "!" or "?"
            elif t == "!":
                punctuation.append(t)
                exclamation_mark_score += 1

            elif t == "?":
                punctuation.append(t)
                question_mark_score += 1

            #########
            # Check insults from wordlist
            if t in insult_list:
                insult_item_count += 1

            # Add text if it is not LBR
            if t != "|LBR|":
                clean.append(lemma.lower())

        # [.. "!", "!", "hallo", "?", "!", ...]
        # Merkel muss weg! rafft ihr es noch?!?!?!?
        # @USER gehts noch?!

        # We give every exclamation/question mark a penalty
        # We dont care if they are single or in a group,
        # just the total mass is counting
        punctuation_score = scale_feature(punctuation)

        punctuation_scores.append(punctuation_score)
        exclamation_mark_numbers.append(exclamation_mark_score)
        question_mark_numbers.append(question_mark_score)

        hashtags.append(hashtag_count)
        mentions.append(mention_count)
        training_text_clean.append(" ".join(clean))
        insult_count.append(insult_item_count)

    data_frame = pd.DataFrame(
        {
            "tweets": training_text_clean,
            "hashtags": hashtags,
            "mentions": mentions,
            "punctuation_score": punctuation_scores,
            "emoji_scores": emoji_scores,
            "insult_count": insult_count
        }
    )
    return data_frame


def train():
    global training_text
    global test_text
    global training_label
    global test_label
    training_text, training_label, test_text, test_label = get_splitted_data(0.7)
    print("Get training features...")
    data_frame = get_feature_data(training_text)

    preprocessor = ColumnTransformer(
        [
            ('tweets', TfidfVectorizer(stop_words=stopwords.words("german"), ngram_range=(1,1)), 'tweets'),
            ('scaler', MinMaxScaler(), ["hashtags", "mentions"])
        ],
        remainder='passthrough', verbose_feature_names_out=True, n_jobs=-1)

    text_clf_svm = Pipeline([
        ('preprocessor', preprocessor),
        ('clf-svm', svm.SVC(class_weight=None, C=7, gamma=0.1, kernel="rbf"))
    ])
    ######################################################
    # Adjust parameters properly
    # TODO: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

    #print("Starting GridSearchCV...")
    #start = datetime.now()
    #C_range = [1, 3, 5, 7, 9, 11]
    #gamma_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    #parameters = {
    #    'clf-svm__C': C_range,
    #    'clf-svm__gamma': gamma_range,
    #}

    #gs_clf = GridSearchCV(text_clf_svm, parameters, cv=5, n_jobs=-1)
    #gs_clf = gs_clf.fit(data_frame, training_label)
    #print("Best Score: " + str(gs_clf.best_score_))
    #print("Best Params: \n")
    #print(gs_clf.best_score_)
    #for param_name in sorted(parameters.keys()):
    #    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    #end = datetime.now()
    #duration = end - start
    #print("GridSearchCV duration: " + str(duration))
    #print(
    #    "The best parameters are %s with a score of %0.2f"
    #    % (gs_clf.best_params_, gs_clf.best_score_)
    #)
    #scores = gs_clf.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))
    #plt.figure(figsize=(8, 6))
    #plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    #plt.imshow(
    #    scores,
    #    interpolation="nearest",
    #    cmap=plt.cm.hot
    #)
    #plt.xlabel("gamma")
    #plt.ylabel("C")
    #plt.colorbar()
    #plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    #plt.yticks(np.arange(len(C_range)), C_range)
    #plt.title("Accuracy")
    #plt.show()
    #
    #   Plot the accuracy
    ##########################
    print("Make model fit...")
    start = datetime.now()
    text_clf_svm = text_clf_svm.fit(data_frame, training_label)
    end = datetime.now()
    duration = end - start
    print("Fit duration: " + str(duration))
    return text_clf_svm


def test(clf):
    print("Get testing features...")
    data_frame = get_feature_data(test_text)
    print("Predict...")

    predicted_svm = clf.predict(data_frame)

    print("Support Vector Maschine:\n" + str(np.mean(predicted_svm == test_label)))
    print(metrics.classification_report(test_label, predicted_svm))


if __name__ == '__main__':
    main()