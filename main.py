import time

import spacy
from spacy.tokens import DocBin
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from nltk.corpus import stopwords

from datacleaning import DataCleaning
from datamodeling import DataModeling
from datapreprocessing import DataPreprocessing
from knowledgegraphgeneration import KnowledgeGraphGeneration
from insertintoneo4j import InsertToNeo4J
import warnings

warnings.filterwarnings('ignore')


class Main:
    def __init__(self):
        self.clean_data = DataCleaning()
        self.process_data = DataPreprocessing()
        self.model_data = DataModeling()
        self.generate_kg = KnowledgeGraphGeneration()
        self.neo4j_insertion = InsertToNeo4J()
        self.file_path = 'data/Symptom2Disease.csv'
        self.ner_model_path = 'models/ner-model/model-best'

    @staticmethod
    def read_data(path):
        data = pd.read_csv(path)
        return data

    def main(self, eda=False, train=False, train_ner=False, save=False):
        print("Reading the data from CSV using pandas.\n")
        df = self.read_data(self.file_path)
        if eda is True:
            self.process_data.exploratory_data_analysis(df)
        time.sleep(2)
        print("Encoding the categorical labels into numerical format for training purpose.\n")
        df, encoder = self.process_data.encode_labels(df)
        time.sleep(2)
        print("Cleaning the raw text by removing stopwords, numbers, punctuations and non alphabets.\n")
        cleaned_data = self.clean_data.remove_stopwords(df['text'], visualize=False)
        # print(df['label_encoded'].values.tolist())
        train_data, train_labels, val_data, val_labels, test_data, test_labels = self.process_data.split_data(
            cleaned_data,
            df['label_encoded'].values.tolist())

        # print("++++++++++", train_labels)

        train_labels = np.array(train_labels, dtype=np.int32)
        val_labels = np.array(val_labels, dtype=np.int32)
        test_labels = np.array(test_labels, dtype=np.int32)

        print("Training Data Size:", len(train_data))
        print("Validation Data Size:", len(val_data))
        print("Testing Data Size:", len(test_data))
        print("\n")

        print("Performing future extraction to extract features from text.")
        padded_train_data = self.process_data.feature_extraction(train_data, fit_on_train=True)
        padded_val_data = self.process_data.feature_extraction(val_data, fit_on_train=False)
        padded_test_data = self.process_data.feature_extraction(test_data, fit_on_train=False)

        time.sleep(2)

        if train is True:
            print("Getting the CNN model.")
            cnn_model = self.model_data.cnn_architecture(self.process_data.max_len)

            print("Training the CNN model.")
            cnn_model_history = cnn_model.fit(padded_train_data, train_labels, epochs=30,
                                              validation_data=(padded_val_data, val_labels))

            if save is True:
                self.model_data.plot_graphs(cnn_model_history, 'accuracy', "cnn")
                self.model_data.plot_graphs(cnn_model_history, 'loss', "cnn")

                cnn_model.save("models/cnn_model.h5")

            print("Getting the BiLSTM model.")
            bi_lstm_model = self.model_data.bi_lstm_architecture(self.process_data.max_len)

            print("Training the BiLSTM model.")
            bi_lstm_model_history = bi_lstm_model.fit(padded_train_data, train_labels, epochs=30,
                                                      validation_data=(padded_val_data, val_labels))

            if save is True:
                self.model_data.plot_graphs(bi_lstm_model_history, 'accuracy', "bilstm")
                self.model_data.plot_graphs(bi_lstm_model_history, 'loss', "bilstm")

                bi_lstm_model.save("models/bi_lstm_model.h5")

        else:
            cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
            bi_lstm_model = tf.keras.models.load_model("models/bi_lstm_model.h5")

        val_preds = cnn_model.predict(padded_val_data)
        test_preds = cnn_model.predict(padded_test_data)
        val_class_predictions = np.argmax(val_preds, axis=1)
        test_class_predictions = np.argmax(test_preds, axis=1)
        print(classification_report(encoder.inverse_transform(val_labels),
                                    encoder.inverse_transform(val_class_predictions)))
        print(classification_report(encoder.inverse_transform(test_labels),
                                    encoder.inverse_transform(test_class_predictions)))

        # print("+++++++++++++++++++++")
        val_preds = bi_lstm_model.predict(padded_val_data)
        test_preds = bi_lstm_model.predict(padded_test_data)
        val_class_predictions = np.argmax(val_preds, axis=1)
        test_class_predictions = np.argmax(test_preds, axis=1)
        print(classification_report(encoder.inverse_transform(val_labels),
                                    encoder.inverse_transform(val_class_predictions)))
        print(classification_report(encoder.inverse_transform(test_labels),
                                    encoder.inverse_transform(test_class_predictions)))

        print("Loading the Named Entity Recognition model using Spacy\n")
        time.sleep(3)

        if train_ner is True:
            ner_training_data = self.process_data.prepare_data_to_train_ner(df)
            print("#######################", ner_training_data[0])
            self.model_data.train_ner(ner_training_data, spacy, DocBin)

        ner_model = spacy.load(self.ner_model_path)

        stop_words = set(stopwords.words('english'))
        stop_words.update(["I've", "I'm", "like"])

        print("Extracting the entities and translating them into symptom-[is_linked_to]->disease format\n")
        triples = self.generate_kg.get_triples(df, ner_model)

        triples_df = pd.DataFrame(triples)

        print("Sample:")
        print(triples_df.head())

        print("\n")
        print("Inserting into Neo4J for visualize and querying purpose\n")
        self.neo4j_insertion.main(triples_df)

        print("Research Project Successfully Executed.")


if __name__ == '__main__':
    obj = Main()
    obj.main(train=False, train_ner=False, eda=False, save=False)
