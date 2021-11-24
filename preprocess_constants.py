import csv
import re


def main():
    preprocess_emojis()
    preprocess_insults()


def preprocess_emojis():
    emoji_scores = {}
    path = "constants/Emoji_Sentiment_Data_clean.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for emoji in reader:
            list = emoji[0].split(";")
            emoji_scores[list[0]] = float(list[2])

    return emoji_scores


def preprocess_insults():
    insult_list = []
    path = "data/insults.txt"
    with open(path, "r", encoding="utf-8") as insults:
        lines = insults.readlines()
        for insult in lines:
            insult_list.append(insult.lower().strip())
    insults.close()
    return insult_list


def encode_unicode_to_emoji():
    with open("train/germeval2019.training.emojis.txt", "w", encoding="utf-8") as new_f:
        with open("train/germeval2019.training_subtask1_2_korrigiert.txt", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                new_line = re.sub("<U\+\w+>", replace_with_emoji, line)
                new_f.write(new_line)
            f.close()
        new_f.close()


def replace_with_emoji(unicode_emoji):
    code = unicode_emoji.group()
    return chr(int(code[3:-1], 16))


if __name__ == "__main__":
    encode_unicode_to_emoji()