# coding: utf-8

from SentimentNetwork import SentimentNetwork
import numpy as np

if __name__ == '__main__':

    g = open('dataset/reviews.txt', 'r')
    reviews = list(map(lambda x: x[:-1], g.readlines()))
    g.close()

    g = open('dataset/labels.txt', 'r')
    labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
    g.close()

    ## Creating Neural Net

    sentiment_analysis = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.1)

    print("Training the neural net")
    sentiment_analysis.train(reviews[:-1000], labels[:-1000])

    print("Running and evaluating a single random review")
    chosen_index = np.random.random_integers(0, len(reviews) - 1)

    print(reviews[chosen_index])
    print(sentiment_analysis.run(reviews[chosen_index]))
