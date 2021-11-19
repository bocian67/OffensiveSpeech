## for data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from nltk.corpus import stopwords
from sklearn import svm, metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
## for machine learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

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


def train():
    print("Begin training")
    # Open the training file
    training = open("data/training_text.txt", "r", encoding="utf-8")
    testing = open("data/training_label.txt", "r", encoding="utf-8")
    for line in training:
        text = line.rstrip("\n")
        text = get_pos(text)
        training_text.append(text)

    for line in testing:
        text = line.rstrip("\n")
        training_label.append(text)

    training.close()
    testing.close()

    ## Count vectorizer
    # count_vect = CountVectorizer(stop_words=stopwords.words("german"))
    # X_train_counts = count_vect.fit_transform(training_text)
    ## TD-IDF transformation
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    ## train svm

    # trainedsvm = svm.SVC(kernel='rbf').fit(X_train_tfidf, training_label)
    #
    # text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('german'))),
    #                         ('tfidf', TfidfTransformer()),
    #                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,
    #                                                   n_iter_no_change=5))])

    # Add features

    #data_frame = pd.DataFrame(
    #    {
    #        "tweets": training_text,
    #    }
    #)

    #preprocessor = ColumnTransformer(
    #    [('tweets',
    #      Pipeline(
    #          [
    #              ('vect', CountVectorizer(stop_words=stopwords.words('german'), ngram_range=(1, 3))),
    #              ('tfidf', TfidfTransformer(use_idf=True))
    #          ]), ['tweets']),
    #     ('tags',
    #      OneHotEncoder(handle_unknown="ignore"), ["tags"])],
    #    remainder='drop', verbose_feature_names_out=True)

    text_clf_svm = Pipeline([
        ("vect", CountVectorizer(stop_words=stopwords.words("german"), ngram_range=(1,3))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf-svm', svm.SVC(kernel='rbf', C=10.0, gamma=0.1, class_weight="balanced"))])

    # C_range = np.logspace(-2, 10, 13)
    # gamma_range = np.logspace(-9, 3, 13)

    # parameters = {
    #    'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
    # }

    # gs_clf = GridSearchCV(text_clf_svm, parameters, cv=5, n_jobs=-1)
    # gs_clf = gs_clf.fit(training_text, training_label)

    # print("Best Score: " + str(gs_clf.best_score_))
    # print("Best Params: \n")
    # print(gs_clf.best_score_)
    # for param_name in sorted(parameters.keys()):
    #    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    text_clf_svm = text_clf_svm.fit(training_text, training_label)

    return text_clf_svm


def test(clf):
    training = open("data/testing_text.txt", "r", encoding="utf-8")
    testing = open("data/testing_label.txt", "r", encoding="utf-8")
    for line in training:
        text = line.rstrip("\n")
        test_text.append(text)

    for line in testing:
        text = line.rstrip("\n")
        test_label.append(text)

    training.close()
    testing.close()

    ## Count vectorizer
    # count_vect = CountVectorizer(stop_words=stopwords.words("german"))
    # X_train_counts = count_vect.fit_transform(test_text)
    ## TD-IDF transformation
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # print("Begin test...")
    ##predicted_svm = clf.predict(test_text)

    predicted_svm = clf.predict(test_text)

    print("Support Vector Maschine:\n" + str(np.mean(predicted_svm == test_label)))
    print(metrics.classification_report(test_label, predicted_svm, target_names=categories))


def test_models():
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))

    # Open the training file
    # Open the training file
    training = open("data/training_text.txt", "r", encoding="utf-8")
    testing = open("data/training_label.txt", "r", encoding="utf-8")
    for line in training:
        text = line.rstrip("\n")
        training_text.append(text)

    for line in testing:
        text = line.rstrip("\n")
        training_label.append(text)

    training.close()
    testing.close()

    print("Begin training")

    # Count vectorizer
    count_vect = CountVectorizer(stop_words=stopwords.words("german"))
    X_train_counts = count_vect.fit_transform(training_text)
    # TD-IDF transformation
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    entries = []
    for model in models:
        model_name = model.__class__.__name__
        print("Model: " + model_name)
        accuracies = cross_val_score(model, X_train_tfidf, training_label, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df,
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()


if __name__ == '__main__':
    main()
