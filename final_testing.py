import pickle
import json
import sys

model_location = sys.argv[1]
testing_location = sys.argv[2]
output_location = sys.argv[3]

with open(f'{model_location}/model.pkl', 'rb') as f:
    model = pickle.load(f)

naive_bayes_classifier, count_vectorizer = model

with open(f'{testing_location}','r',encoding='utf-8') as f:
    test_data = json.load(f)

X_testing = count_vectorizer.transform([entry['text'] for entry in test_data])

predictions = naive_bayes_classifier.predict(X_testing)

with open(f'{output_location}','w',encoding='utf-8') as f:
    for i in range(len(test_data)):
        f.write(f'{predictions[i]}\n')
