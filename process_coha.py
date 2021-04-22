"""
File: create-word-set.py
------------------------
Combines words from several sources into a document to create a
robust english language dictionary.
"""
import os
import re
import string
from settings import ENGLISH_LEXICON

COHA_DIRECTORY = "./COHA-sample-data/"
GOOGLE_FILE = "./google-10000-english-no-swears.txt"

def main():
    word_set = set()
    # adding all the COHA data
    for f in os.listdir(COHA_DIRECTORY):
        textfile = open(f"./{COHA_DIRECTORY}/{f}")
        print(f"Processing File: {f}")
        for line in textfile:
            word = clean_text_v2(line.split("\t")[1])
            if word.isalpha():
                word_set.add(word)
        print(f"{f} file processed.")

    # adding all the google data
    for line in open(GOOGLE_FILE):
        word_set.add(clean_text_v2(line))
    print("Google Data Processed.")

    new_doc = open(ENGLISH_LEXICON, "w+")
    count = 0
    for word in word_set:
        # space is added to make it easier to find specific words in the file
        new_doc.write(" " + word + " "+ "\n")
        count += 1
    new_doc.close()
    print("Text written to file.")
    print(f"Total Words: {count}")


def clean_text_v2(text):
    text = text.lower()
    text.strip("")
    text = re.sub('\n', "", text)
    text = re.sub('\t', "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    return text

if __name__ == "__main__":
    main()