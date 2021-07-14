"""
File: times_process.py
----------------------
Extracts text from the times-random-sample folder for testing
"""
import os
import glob
import numpy as np
import pandas as pd
import openpyxl
import pathlib
from process_coha import clean_text_v2


TEXT_DIRECTORY = "./text-to-predict/raw-text/"
TIMES_DIRECTORY = "./times-random-sample/"

def main():
    for xl_file in glob.glob(os.path.join(TIMES_DIRECTORY, '*.xlsx')):
        df = pd.read_excel(xl_file, engine = 'openpyxl')
        pathlib.Path(TEXT_DIRECTORY + str(df.iloc[1]["file_name"])).mkdir(parents=True, exist_ok=True)
        df.apply(lambda row: sort_data(row),axis=1,result_type='expand')


def sort_data(row: pd.Series) -> None:
    if row["category"] == "News":
        filename = clean_text_v2("".join(row["title"].split()[0:3])) + ".txt"
        text_file = open(TEXT_DIRECTORY + row["file_name"] + "/" + filename, "w+")

        text_file.write(clean_text_v2(str(row["text"])))

        text_file.close()
        print("Created File: " + filename)


if __name__ == "__main__":
    main()