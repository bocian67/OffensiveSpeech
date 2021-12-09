import random

training_text = []
training_label = []

testing_text = []
testing_label = []

other_text = []
insult_text = []
profanity_text = []
abusive_text = []

sample_count = 0


def add_text_to_classified_data(f):
    file_lines = f.readlines()
    # Extract text and insult classification for each line
    for line in file_lines:
        splitted_line = line.split("\t")
        text = splitted_line[0]
        classification = splitted_line[2].rstrip()
        if classification == "OTHER":
            other_text.append(text)
        elif classification == "ABUSE":
            abusive_text.append(text)
        elif classification == "INSULT":
            insult_text.append(text)
        elif classification == "PROFANITY":
            profanity_text.append(text)


def get_data():
    global sample_count
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
        add_text_to_classified_data(f)
        f.close()

    sample_count = len(profanity_text) + len(other_text) + len(insult_text) + len(abusive_text)
    print("[*] Having " + str(sample_count) + " samples")


# training_part: Percentage of each classified tweet which should be in training
# Example: 0.7
def get_splitted_data(training_part):
    global training_text
    global training_label
    global testing_text
    global testing_label

    # Obtain data
    get_data()

    # Get count of training samples
    profanity_training_count = int(len(profanity_text) * training_part)
    insult_training_count = int(len(insult_text) * training_part)
    abusive_training_count = int(len(abusive_text) * training_part)
    other_training_count = int(len(other_text) * training_part)

    # Get count of testing samples
    profanity_testing_count = int(len(profanity_text) - profanity_training_count)
    insult_testing_count = int(len(insult_text) - insult_training_count)
    abusive_testing_count = int(len(abusive_text) - abusive_training_count)
    other_testing_count = int(len(other_text) - other_training_count)

    # Shuffle order of tweets
    random.shuffle(profanity_text)
    random.shuffle(insult_text)
    random.shuffle(abusive_text)
    random.shuffle(other_text)

    # Add training text and label
    training_text += profanity_text[:profanity_training_count]
    training_label += ["PROFANITY"] * profanity_training_count

    training_text += insult_text[:insult_training_count]
    training_label += ["INSULT"] * insult_training_count

    training_text += abusive_text[:abusive_training_count]
    training_label += ["ABUSE"] * abusive_training_count

    training_text += other_text[:other_training_count]
    training_label += ["OTHER"] * other_training_count

    # Add testing text and label
    testing_text += profanity_text[profanity_training_count:]
    testing_label += ["PROFANITY"] * profanity_testing_count

    testing_text += insult_text[insult_training_count:]
    testing_label += ["INSULT"] * insult_testing_count

    testing_text += abusive_text[abusive_training_count:]
    testing_label += ["ABUSE"] * abusive_testing_count

    testing_text += other_text[other_training_count:]
    testing_label += ["OTHER"] * other_testing_count

    print("Profanity: " + str(len(profanity_text)))
    print("Insult: " + str(len(insult_text)))
    print("Abuse: " + str(len(abusive_text)))
    print("Others: " + str(len(other_text)))

    print("[*] Having " + str(len(training_text)) + " TRAINING samples")
    print("[*] Having " + str(len(testing_text)) + " TESTING samples")

    return training_text, training_label, testing_text, testing_label









