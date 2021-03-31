import numpy as np
import tensorflow as tf
from tensorflow import keras
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def main():
    graph = load_pb()
    input = graph.get_tensor_by_name('input:0')
    print(input)
    


def test():
    tokens = word_tokenize("The quick brown fox jumps over the lazy dog")
    print(tokens)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]
    print(tokens)
    porter = PorterStemmer()
    stems = []
    for t in tokens:
        stems.append(porter.stem(t))
    print(stems)


def load_pb():
    with tf.io.gfile.GFile("./s2s/saved_model.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

if __name__ == "__main__":
    main()
