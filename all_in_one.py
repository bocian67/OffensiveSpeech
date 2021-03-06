import emojis
import pandas as pd
import spacy
from matplotlib.colors import Normalize
from nltk.corpus import stopwords
from sklearn import svm, metrics
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from evaluation import *
from get_dataset import get_splitted_data
from preprocess_constants import *

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


def get_feature_data(text_collection):
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
    for text in text_collection:
        # spaCy lemma
        # sp??ter pipeline durch aufrufe ??ndern -> performance
        doc = nlp(text)
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
        emoji_iter = emojis.iter(text)
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

        cleaned_input_text = " ".join(clean)

        hashtags.append(hashtag_count)
        mentions.append(mention_count)
        training_text_clean.append(cleaned_input_text)
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
        ('clf-svm', svm.SVC(class_weight=None, C=6, gamma=0.15, kernel="rbf"))
    ])


    ###
    # Gridsearch options
    #C_range = [2, 4, 6, 8, 10, 14, 20]
    #gamma_range = [0.05, 0.1, 0.15, 0.2, 0.4, 1, 2, 5]
    #parameters = {
    #    'clf-svm__C': C_range,
    #    'clf-svm__gamma': gamma_range,
    #}
    #search_for_parameters(text_clf_svm, data_frame, training_label, parameters)

    ###
    # Plot GridSearch options
    #get_plot(gamma_range, C_range)

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