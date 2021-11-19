import csv


def main():
    preprocess_emojis()


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


if __name__ == "__main__":
    main()