import json


def load_movies() -> list:
    with open("./data/movies.json") as file:
        return json.load(file)["movies"]


def load_stopwords() -> list:
    with open("./data/stopwords.txt") as file:
        return file.read().splitlines()


def load_golden_dataset() -> list:
    with open("./data/golden_dataset.json") as file:
        return json.load(file)["test_cases"]
