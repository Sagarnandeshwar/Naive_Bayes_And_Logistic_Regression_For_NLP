# Naive Bayes And Logistic Regression For NLP
We implement Naive Bayes and K-fold cross-validation from scratch, while using logistic (softmax) regression and compare these two algorithms on two distinct textual datasets. 

## Naive Bayes Classifier 
Naive Bayes is a generative classifierer which assumes that the features are conditionally independent which reduces the number of parameters required. This main idea is tofind the probabilities of categories given a text document by using the joint probabilities of words and categories 

## Logistic Regression 
Logistic regression is a classifier which uses a sigmoid function with a cross entropy cost to find a linear decision boundary. 

## Dataset 
### TwentyNewsGroup 
The TwentyNewsGroup Dataset consists of close to 18000 newsgroups posts split on 20 different topics. It is divided into two subsets: one for training and the other one for testing. The split between the train and test set is based on whether a post was made before or after a specific date.  

### Sentiment140 
The Sentiment140 dataset is a CSV that consists of various tweets with emoticons removed. There are 6 fields in the data; the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive), the tweet id, the date the tweet was made, the query, the user that tweeted it, and the content of the tweet [2]. For our application, we only considered binary classification (0 = negative and 4 = positive) for the project. 

## Featurev Engineering  
### Count Vectorizer 
We start with the text data and convert text to feature vectors. 
### Cleaning dataset 
1. Removing headers, footers and quotes: Use the default train subset (subset=‘train’, and remove=([‘headers’, ‘footers’, ‘quotes’]) in sklearn.datasets) to train the models and report the final performance on the test subset 
2. Pre-Processing: We removed html tags, punctuation, brackets, non-ASCII characters and digits from our data. 

### TF-IDF Vectorizer 
This method uses word’s frequency(TF) and the inverse document frequency( IDF). Each word is given a TF score and an IDF score and then the method weight of the word is the product of the scores. The term frequency is just the frequency of the word in the document. The IDF score is the log of the ratio of total number of documents and the number of documents which contain the word. Therefore, rarer words will have a higher score. 

## Result  

From the result, we can see that Naive Bayes performed well overall. However, a more accurate assessment and a better evaluation would have been made, if we would have tweaked the following parameters if we had not experienced limited computing resources: 
- Compared efficient feature extraction using CountVectorizer, tf-idf matrix, CounterVectorizer bigram and chose the optimum feature extraction method. 
- Used greater training model for Sentiment140 (6% taken) and more classes for TwentyNewsGroups (4 out of 20 were taken) datasets. However, with initial assessment, it was pointed out that if all classes for TwentyNewsGroups were taken then accuracy drops down to 50%. A more assessment was needed however, this consumes an enormous amount of time and running K-fold validation with large amount of classes or testing sample was inadequate. 
- Confusion matrix for both models on dataset would have been plotted to understand the true/false positive by the model. 
- More hyperparameters for softmax regression could have been explored like solver, number of epochs, fit intercept etc. We only norm, and regularization coefficient for this project. 
- Increased the number of hyperparameters and number of K-folds for better hyperparameter tuning. 
