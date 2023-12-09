from datapreprocessing import DataPreprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from spacy.util import filter_spans
import os
import time
import subprocess


class DataModeling:
    def __init__(self):
        self.process_data = DataPreprocessing()

    @staticmethod
    def plot_graphs(history, metric, flag):
        plt.figure(figsize=(15, 8))
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(history.history[metric])
        ax.plot(history.history['val_' + metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([metric, 'val_' + metric])
        # plt.title(metric)
        # plt.show()
        plt.savefig("plots/" + flag + "_" + metric + ".png", dpi=300)

    def cnn_architecture(self, max_length):
        cnn_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(2000, 128, input_length=max_length),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(24, activation='softmax')])

        print(cnn_model.summary())

        cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return cnn_model

    @staticmethod
    def bi_lstm_architecture(max_length):
        tf.random.set_seed(42)
        lstm_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(2000, 128, input_length=max_length),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(24, activation='softmax')])

        print(lstm_model.summary())

        lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return lstm_model

    @staticmethod
    def train_ner(TRAIN_DATA, spacy, DocBin):
        nlp = spacy.blank("en")
        doc_bin = DocBin()

        for training_example in tqdm(TRAIN_DATA):
            text = training_example['text']
            labels = training_example['entities']
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in labels:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    print("Skipping entity")
                else:
                    ents.append(span)
            filtered_ents = filter_spans(ents)
            doc.ents = filtered_ents
            doc_bin.add(doc)

        doc_bin.to_disk("data/training_data.spacy")
        command = "python -m spacy train config.cfg --output ./models/ner-model --paths.train ./data/training_data.spacy " \
                  "--paths.dev ./data/training_data.spacy"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Command executed successfully.")
        else:
            print(f"Error: {result.stderr}")

        time.sleep(2)
