"""
This was a program in one of the earlier drafts of the model that processed the data generated from 
hand-corrected OCR'ed documents.
"""


def main():
    origfile = open("TRAINING-DATA/corrected-data.txt")
    newfile = open("letter-conversions.txt", "w+")
    for line in origfile:
        line = line.split("\t")
        bad_word = line[0].lower()
        correct_word = line[1].lower()

        len_limit = len(bad_word) if len(bad_word) < len(correct_word) else len(correct_word)
        print(bad_word, correct_word)
        while (i < len_limit):
            if bad_word[i] != correct_word[i]:
                break
            i += 1
        while (j < len_limit):
            if bad_word[len(bad_word) - j] != correct_word[len(correct_word) - j]:
                break
            j += 1
        
        #print(line, i,j)
        d = {}
        bad_part, correct_part = "", ""
        for k in range(i, len_limit - j + 1):
            try:
                bad_part += bad_word[k]
                correct_part += correct_word[k]
            except:
                pass
        
        for i in range(len(bad_part)):
            try:
                newfile.write(f"{correct_part[i]}\t{bad_part[i]}\n")
            except:
                newfile.write(f"{correct_part}\t{bad_part}\n")
    newfile.close()

            







if __name__ == "__main__":
    main()