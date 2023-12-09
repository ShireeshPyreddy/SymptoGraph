import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import re

from datacleaning import DataCleaning

sns.set_palette('Set2')


class DataPreprocessing:
    def __init__(self):
        self.clean_data = DataCleaning()
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
        self.max_len = 50

    @staticmethod
    def get_word_cloud(data):
        stop_words = set(stopwords.words('english'))
        stop_words.update(["I've", "I'm", "like"])

        wordcloud = WordCloud(width=800,
                              height=800,
                              background_color='white',
                              stopwords=stop_words,
                              min_font_size=8).generate(str(data['text']))

        return wordcloud

    def exploratory_data_analysis(self, data):
        # print(data['label'].value_counts())
        # print(len(data['label'].value_counts()))
        text_length = data['text'].apply(lambda x: len(x.split(' ')))
        plt.figure(figsize=(15, 8))
        sns.histplot(text_length)
        plt.xlabel('Length of Text')
        # plt.show()
        plt.savefig("plots/text_length.png", dpi=300)

        wordcloud = self.get_word_cloud(data)

        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        # plt.show()
        plt.savefig("plots/frequently-occured-words.png", dpi=300)

    @staticmethod
    def encode_labels(data):
        def label_encode(labels):
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            label_sequences = label_encoder.transform(labels)
            return label_sequences, label_encoder

        data['label_encoded'], encoder = label_encode(data['label'])

        return data, encoder

    @staticmethod
    def split_data(data, labels):
        print("Splitting the data into training, validation and test set")
        train_data, val_data, train_labels, val_labels = train_test_split(data,
                                                                          labels,
                                                                          test_size=0.20,
                                                                          random_state=42)

        val_data, test_data, val_labels, test_labels = train_test_split(val_data,
                                                                        val_labels,
                                                                        test_size=0.50,
                                                                        random_state=42)

        return train_data, train_labels, val_data, val_labels, test_data, test_labels

    def feature_extraction(self, data, fit_on_train):
        if fit_on_train is True:
            self.tokenizer.fit_on_texts(data)
        sequences = self.tokenizer.texts_to_sequences(data)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_len, padding='post')
        return padded_sequences

    @staticmethod
    def prepare_data_to_train_ner(data):
        with open("ners.txt", 'r') as sym:
            ners = sym.read().splitlines()

        ner_training_data = []
        for each_row in data.iterrows():
            text = each_row[1]['text']
            temp = []
            occurrences = [(start, start + len(keyword), keyword) for keyword in set(ners) for start in
                           (i.start() for i in re.finditer(r'\b' + re.escape(keyword.lower()) + r'\b', text.lower()))]

            # print(text, occurrences)

            ner_training_data.append({'entities': occurrences, 'text': text})

        return ner_training_data
