import spacy

nlp = spacy.load("de_core_news_lg")
categories = ["OTHER", "INSULT", "ABUSE", "PROFANITY"]

def write_text():
    training_text = []
    training_label = []
    # Open the training file
    with open("train/germeval2018.training.txt", encoding='utf-8') as f:
        file_lines = f.readlines()
        # Extract text and insult classification for each line
        training = open("data/training_text.txt", "w", encoding="utf-8")
        testing = open("data/training_label.txt", "w", encoding="utf-8")
        for line in file_lines:
            splitted_line = line.split("\t")
            text = splitted_line[0]
            classification = splitted_line[2].rstrip()
            if classification != "OTHER":
                cleaned_text = clean_text(text)
                training.write(cleaned_text + "\n")
                testing.write(classification + "\n")
        f.close()

    with open("train/germeval2018.test_.txt", encoding="utf-8") as f:
        file_lines = f.readlines()
        # Extract text and insult classification for each line
        training = open("data/testing_text.txt", "w", encoding="utf-8")
        testing = open("data/testing_label.txt", "w", encoding="utf-8")
        for line in file_lines:
            splitted_line = line.split("\t")
            text = splitted_line[0]
            classification = splitted_line[2].rstrip()
            if classification != "OTHER":
                cleaned_text = clean_text(text)
                training.write(cleaned_text + "\n")
                testing.write(classification + "\n")
        f.close()


def clean_text(text):
    # spaCy lemma
    # später pipeline durch aufrufe ändern -> performance
    doc = nlp(text)
    clean = []
    for token in doc:
        word = token.lemma_
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


if __name__ == '__main__':
    write_text()
