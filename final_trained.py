import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
import sys


def model():
    file_location = sys.argv[1]
    model_location = sys.argv[2]
    with open(f'{file_location}/train.json', 'r',encoding='utf-8') as f:
        data_train = json.load(f)
    with open(f'{file_location}/valid.json', 'r',encoding='utf-8') as f:
        data_val = json.load(f)


    # Extract sentences and languages
    sentences_train = [entry['text'] for entry in data_train]
    languages_train = [entry['langid'] for entry in data_train]
    sentences_val = [entry['text'] for entry in data_val]
    languages_val = [entry['langid'] for entry in data_val]

    # Step 2: Split data into training and testing sets
    X_train, Y_train = sentences_train+sentences_val, languages_train+languages_val

    del(sentences_train)
    del(languages_train)
    del(sentences_val)
    del(languages_val)

    for i in range(0,1):
        X_test=X_train
        Y_test=Y_train
        count_vectorizer = CountVectorizer(ngram_range=(4,6),analyzer='char')
        X_train_counts = count_vectorizer.fit_transform(X_train)
        X_test_counts = count_vectorizer.transform(X_test)
        naive_bayes_classifier = MultinomialNB(alpha=0.01)
        naive_bayes_classifier.fit(X_train_counts, Y_train)
        predictions_nb = naive_bayes_classifier.predict(X_test_counts)
        X_train_new=[]
        Y_train_new=[]
        for i in range(len(X_train)):
            if Y_train[i]==predictions_nb[i]:
                X_train_new.append(X_train[i])
                Y_train_new.append(Y_train[i])
        X_train=X_train_new
        Y_train=Y_train_new
        del(X_train_new)
        del(Y_train_new)
        del(count_vectorizer)
        del(X_train_counts)
        del(X_test_counts)
        del(naive_bayes_classifier)
        del(predictions_nb)

    print("Data cleaned")

    X_train_new = []
    Y_train_new = []

    for i in range(len(X_train)):
        if Y_train[i] == 'ta' or Y_train[i]=='kn' or Y_train[i]=='ml' or Y_train[i]=='hi' or Y_train[i]=='bn' or Y_train[i]=='mr':
            for j in range(17):
                X_train_new.append(X_train[i])
                Y_train_new.append(Y_train[i])
        else:
            X_train_new.append(X_train[i])
            Y_train_new.append(Y_train[i])

    X_train = X_train_new
    Y_train = Y_train_new

    del(X_train_new)
    del(Y_train_new)

    print("Start vectorize")

    count_vectorizer_1 = CountVectorizer(ngram_range=(4,6),analyzer='char')
    count_vectorizer_2 = CountVectorizer(ngram_range=(2,2),analyzer='word')
    count_vectorizer = FeatureUnion([
        ('count1', count_vectorizer_1),
        ('count2', count_vectorizer_2)
    ]) 
    # count_vectorizer = CountVectorizer(ngram_range=(4,6), analyzer='char')
    X_train_counts = count_vectorizer.fit_transform(X_train)

    print("Start NB")
    naive_bayes_classifier = MultinomialNB(alpha=0.01)
    naive_bayes_classifier.fit(X_train_counts, Y_train)

    pickle.dump((naive_bayes_classifier, count_vectorizer), open(f'{model_location}/model.pkl', 'wb'))

if __name__ == "__main__":
    model()
        