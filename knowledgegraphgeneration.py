import pandas as pd
from formpharses import generate_phrases
from nltk.corpus import stopwords
import nltk
from openie import StanfordOpenIE


class KnowledgeGraphGeneration:

    def __init__(self):
        self.properties = {
            'openie.affinity_probability_cap': 2 / 3,
        }

    def filter(self, triples):
        filtered_triples = []
        for index, each in enumerate(triples):
            obj = each['object']
            flag = False
            for next_index, next_each in enumerate(triples):
                if index != next_index:
                    if obj in next_each['object'] or len(
                            set(obj.split()).intersection(set(next_each['object'].split()))) == len(obj.split()):
                        flag = True
                        break
            if flag is False:
                filtered_triples.append(each)

        return filtered_triples

    @staticmethod
    def map_pos_phrases(pos_tags, phrases):
        mappings = []
        for each in phrases:
            temp = []
            for each_word in each.split():
                for each_tag in pos_tags:
                    if each_word in each_tag or each_word.rstrip(".").strip() in each_tag:
                        temp.append(each_tag[-1])
                        break
            mappings.append(" ".join(temp))

        return mappings

    def get_triples1(self, data, stop_words):
        data_ = []
        for i in data.iterrows():
            # print("+++++++++")
            # print(i[1]['text'])
            tokens = nltk.word_tokenize(i[1]['text'])
            tag = nltk.pos_tag(tokens)
            print(tag)
            phrases = generate_phrases(i[1]['text'], stop_words)
            print(phrases)

            mapped_pos_tags = self.map_pos_phrases(tag, phrases)
            print(mapped_pos_tags)

            # triples = []
            # with StanfordOpenIE(properties=properties) as client:
            #     for triple in client.annotate(i[1]['text']):
            #         print('|-', triple)
            #         if triple['object'] not in stop_words:
            #             triples.append(triple)

            # filtered_triples = filter(triples)
            filtered_triples = []

            for ph, map_pos in zip(phrases, mapped_pos_tags):
                if map_pos in ["JJ NN", "JJ NNS"]:
                    filtered_triples.append(
                        {"subject": ph.rstrip("."), "relation": "is linked to", "object": i[1]['label'].lower()})

            print("########")

            for each in filtered_triples:
                print(each)
                each['subject'] = each['subject'].lower()
                each['relation'] = each['relation'].lower()
                each['object'] = each['object'].lower()
                data_.append(each)

        return data_

    @staticmethod
    def get_triples(data, model):

        print("Loading the NERs data from a text file instead of passing it directly to the model to speed up the "
              "execution process")

        with open("ners.txt", 'r') as sym:
            ners = sym.read().splitlines()

        triples = []
        for i in data.iterrows():
            # print("+++++++++")
            # print(i[1]['text'])
            for ner in ners:
                if ner.lower().replace("_", " ") in i[1]['text'].lower():
                    triples.append(
                        {"Symptom": ner.lower(), "Relation": "is_linked_to", "Disease": i[1]['label'].lower()})

        return triples
