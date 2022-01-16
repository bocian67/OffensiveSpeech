training_text = []
training_label = []

testing_text = []
testing_label = []

def add_text_to_classified_test(f):
    file_lines = f.readlines()
    # Extract text and insult classification for each line
    training_count = {
        "INSULT": 0,
        "ABUSE": 0,
        "PROFANITY": 0,
        "OTHER": 0

    }
    for line in file_lines:
        splitted_line = line.split("\t")
        text = splitted_line[0]
        classification = splitted_line[2].rstrip()
        testing_text.append(text)
        testing_label.append(classification)
        training_count[classification] += 1
    print(f.name)
    print(training_count)


def add_text_to_classified_data(f):
    file_lines = f.readlines()
    testing_count = {
        "INSULT": 0,
        "ABUSE": 0,
        "PROFANITY": 0,
        "OTHER": 0

    }
    # Extract text and insult classification for each line
    for line in file_lines:
        splitted_line = line.split("\t")
        text = splitted_line[0]
        classification = splitted_line[2].rstrip()
        training_text.append(text)
        training_label.append(classification)
        testing_count[classification] += 1
    print(f.name)
    print(testing_count)



def get_data():
    global training_text
    global training_label
    global testing_text
    global testing_label

    print("Obtain data...")
    with open("train/germeval2018.training.txt", encoding='utf-8') as f:
        add_text_to_classified_data(f)
        f.close()
    with open("train/germeval2019.training.emojis.txt", encoding='utf-8') as f:
        add_text_to_classified_data(f)
        f.close()
    with open("train/germeval2018.test_.txt", encoding="utf-8") as f:
        add_text_to_classified_data(f)
        f.close()
    with open("train/testdaten_2019.txt", encoding="utf-8") as f:
        add_text_to_classified_test(f)
        f.close()

    print("[*] Having " + str(len(training_text)) + " TRAINING samples")
    print("[*] Having " + str(len(testing_text)) + " TESTING samples")

    return training_text, training_label, testing_text, testing_label