import emojis
import pandas as pd
import spacy
import os
from imblearn.over_sampling import SVMSMOTE
from nltk import download
from nltk.corpus import stopwords
from sklearn import svm, metrics
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import evaluation
from evaluation import *
from get_dataset import get_data
from preprocess_constants import *

download('stopwords')

nlp = spacy.load("de_core_news_lg")

## Constants
# This should be our data
categories = ["INSULT", "ABUSE", "PROFANITY", "OTHER"]

# Scores
punctuation_score_dict = {
    "!": 1,
    "?": 1,
    "default_emoji": 0
}

training_text = []
training_label = []
test_text = []
test_label = []
insult_list = []

label_encoder = LabelEncoder()

svm_C = 8
svm_gamma = 0.4

preprocessor = ColumnTransformer(
        [
            ('tweets', TfidfVectorizer(stop_words=stopwords.words("german"), ngram_range=(1, 1)), 'tweets'),
            ('scaler', MinMaxScaler(), ["hashtags", "mentions"])
        ],
        remainder='passthrough', verbose_feature_names_out=True, n_jobs=-1)

svm_model = Pipeline([
        ('clf-svm', svm.SVC(class_weight=None, C=svm_C, gamma=svm_gamma, kernel="rbf"))
    ])


def main():
    start_time = datetime.now()
    global insult_list
    insult_list = preprocess_insults()
    emoji_scores = preprocess_emojis()
    punctuation_score_dict.update(emoji_scores)
    print("Run mit C="+str(svm_C)+" und gamma="+str(svm_gamma))
    # Train the model
    train()
    # Test the model
    test()
    end_time = datetime.now()
    duration = end_time - start_time
    print("Full Run Duration: " + str(duration))


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

        clean_text = " ".join(clean)

        punctuation_score = scale_feature(punctuation)
        punctuation_scores.append(punctuation_score)
        exclamation_mark_numbers.append(exclamation_mark_score)
        question_mark_numbers.append(question_mark_score)
        hashtags.append(hashtag_count)
        mentions.append(mention_count)
        training_text_clean.append(clean_text)
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
    global preprocessor
    global svm_model

    training_text, training_label, test_text, test_label = get_data()
    print("Get training features...")

    data_frame = get_feature_data(training_text)
    pre_data = preprocessor.fit_transform(data_frame)
    encoded_labels = label_encoder.fit_transform(training_label)
    mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    print(mapping)

    strategy = {1: 2500, 3: 1000}
    oversample = SVMSMOTE(sampling_strategy=strategy, n_jobs=-1)
    oversampled_data, oversampled_label = oversample.fit_resample(pre_data, encoded_labels)

    print("Make model fit...")
    start = datetime.now()
    svm_model = svm_model.fit(oversampled_data, oversampled_label)
    end = datetime.now()
    duration = end - start
    print("Fit duration: " + str(duration))


def test():
    global svm_model
    global preprocessor
    global test_text

    print("Get testing features...")
    data_frame = get_feature_data(test_text)
    print("Predict...")

    fitted_data = preprocessor.transform(data_frame)

    encoded_label = label_encoder.transform(test_label)
    mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
    predicted_svm = svm_model.predict(fitted_data)

    eval_path = "evaluation"
    exists_eval_path = os.path.exists(eval_path)

    if not exists_eval_path:
        os.makedirs(eval_path)

    with open("evaluation/Leichtmatrosen_fine_2.txt", encoding="utf-8", mode="w") as f:
        for i in range(len(predicted_svm)):
            output = test_text[i] + "\t" + mapping[predicted_svm[i]] + "\n"
            print(output)
            f.write(output)

    print("Support Vector Maschine:\n" + str(np.mean(predicted_svm == encoded_label)))
    print(metrics.classification_report(encoded_label, predicted_svm))

    evaluation.evaluateModelAccuracy(predicted_svm, encoded_label, mapping)


if __name__ == '__main__':
    main()