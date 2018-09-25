# Character level Sentiment models

## Requirements:

- pandas 0.20.3

- tensorflow 1.4.0

- keras 2.0.8

- numpy 1.14.0

These models are based on Karpathy's blog on the The Unreasonable Effectiveness of Recurrent Neural Networks and Christopher Olah's blog on Understanding LSTMS <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>

## DataSet:

I have used the IMDB Movies dataset from Kaggle https://www.kaggle.com/c/word2vec-nlp-tutorial/data, labeledTrainData.tsv which contains 25000 reviews with labels


### Preprocessing on the Data:
I have done minimal preprocessing on the input reviews in the dataset following these basic steps:
1. Remove html tags
2. Replace non-ascii characters with a single space
3. Split each review into sentences

Then I create the character set with a max sentence length of 512 chars and set an upper bound of 15 for the max number of sentences per review. 
The input X is indexed as (document, sentence, char) and the target y has the corresponding sentiments.

