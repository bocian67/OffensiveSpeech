## for data
import numpy as np
from nltk import download
from nltk.corpus import stopwords
## for machine learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import spacy

## for plotting
## for statistical tests
download('stopwords')

# WICHTIG: download im terminal mit:
# python -m spacy download de_core_news_lg
nlp = spacy.load("de_core_news_lg")

## for explainer

# This should be our data
categories = ["OTHER", "INSULT", "ABUSE", "PROFANITY"]


"""
We performed minimal pre-processing by:
• replacing all mentions/usernames with the
generic form User;
• removing the line break characters |LBR|;
• removing the hash character from all hashtags;
(• remove emojis)
• removing stop words using the Python module
stop-words

SVM weight!
"""


def main():
    # Train the model
    clf = train()
    # Test the model
    test(clf)


def clean_text(text):
    # spaCy lemma
    # später pipeline durch aufrufe ändern -> performance
    doc = nlp(text)
    clean = []
    for token in doc:
        word = token.lemma_
        #print("Text:\t" + token.text)
        #print("lemma:\t" + word)
        # Remove mentions & Remove line breaks
        if not ((word.__contains__("@")) or (word == "|LBR|") or (word.startswith("#"))):
            clean.append(word)
            # Remove hashtags
        if word.startswith("#"):
            clean.append(word[1:])

    # Display spaCy content in web browser
    # spacy.displacy.serve(doc, style="dep")

    tweet = " ".join(clean)
    return tweet


def train():
    print("Begin training")
    training_text = []
    training_label = []
    # Open the training file
    with open("train/germeval2018.training.txt", encoding='utf-8') as f:
        file_lines = f.readlines()
        # Extract text and insult classification for each line
        other = 0
        abuse = 0
        insult = 0
        profanity = 0
        for line in file_lines:
            splitted_line = line.split("\t")
            text = splitted_line[0]
            cleaned_text = clean_text(text)
            training_text.append(cleaned_text)
            classification = splitted_line[2].rstrip()
            if classification == "INSULT":
                insult += 1
            elif classification == "OTHER":
                other += 1
            elif classification == "ABUSE":
                abuse += 1
            elif classification == "PROFANITY":
                profanity += 1
            training_label.append(classification)

        f.close()
        print("Other: " + str(other))
        print("Insult: " + str(insult))
        print("Profanity: " + str(profanity))
        print("Abuse: " + str(abuse))

    text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words=stopwords.words('german'))), ('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,
                                                       n_iter_no_change=5))])

    text_clf_svm = text_clf_svm.fit(training_text, training_label)

    return text_clf_svm


def test(clf):
    test_text = []
    test_label = []
    with open("train/germeval2018.test_.txt", encoding="utf-8") as f:
        file_lines = f.readlines()
        # Extract text and insult classification for each line
        other = 0
        abuse = 0
        insult = 0
        profanity = 0
        for line in file_lines:
            splitted_line = line.split("\t")
            test_text.append(splitted_line[0])
            classification = splitted_line[2].rstrip()
            if classification == "INSULT":
                insult += 1
            elif classification == "OTHER":
                other += 1
            elif classification == "ABUSE":
                abuse += 1
            elif classification == "PROFANITY":
                profanity += 1
            test_label.append(classification)

        f.close()
        print("\nOther: " + str(other))
        print("Insult: " + str(insult))
        print("Profanity: " + str(profanity))
        print("Abuse: " + str(abuse))
    print("Begin test...")
    predicted_svm = clf.predict(test_text)

    other = 0
    abuse = 0
    insult = 0
    profanity = 0
    print("\nPredicted:")
    for type in predicted_svm:
        if type == "INSULT":
            insult += 1
        elif type == "OTHER":
            other += 1
        elif type == "ABUSE":
            abuse += 1
        elif type == "PROFANITY":
            profanity += 1

    print("Other: " + str(other))
    print("Insult: " + str(insult))
    print("Profanity: " + str(profanity))
    print("Abuse: " + str(abuse))

    print("Support Vector Maschine:\n" + str(np.mean(predicted_svm == test_label)))


if __name__ == '__main__':
    main()
