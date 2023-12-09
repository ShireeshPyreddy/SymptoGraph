import re

char_splitter = re.compile("[,;!:()|]")


def remove_special_characters(text):
    return re.sub("([{}@\"$%\\\*'\"])", "", text)


def generate_phrases(text, stopwords):
    text = remove_special_characters(text)

    text = " ".join([t if t.isupper() and len(t) > 1 else t.lower() for t in text.split()])
    split_text = char_splitter.split(text)

    phrases = []

    for each_split_text in split_text:
        temp = []
        words = re.split("\\s+", each_split_text)
        previous_stop = False

        for w in words:

            if w in stopwords and not previous_stop:
                temp.append(";")
                previous_stop = True
            elif w not in stopwords:
                temp.append(w.strip())
                previous_stop = False
        temp.append(";")
        phrases.extend(temp)

    final_phrases = re.split(";+", ' '.join(phrases))

    final_phrases = [p.strip() for p in final_phrases]

    return final_phrases