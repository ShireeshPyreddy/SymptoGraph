import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


class DataCleaning:

    @staticmethod
    def remove_stopwords(texts, visualize=False):
        # first remove all punctuations
        words = [re.split(r'[ ,.]+', text) for text in texts]

        # remove space and make all words lower case
        words = [[word.lower() for word in text if word != ''] for text in words]

        # remove stop words using nltk (or you can use your own stop words, see below)
        stop_words = set(stopwords.words('english'))

        words = [[word for word in text if word.lower() not in stop_words] for text in words]

        if visualize is True:
            words_length = [len(w) for w in words]
            plt.hist(words_length, bins=20)
            plt.xlabel('Length of words')
            plt.show()

        return words
