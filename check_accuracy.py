import os
import sys
import json
from sklearn.metrics import accuracy_score

# read output.txt
with open('output.txt', 'r') as f:
    y_pred = f.readlines()
    # remove \n from each line
    y_pred = [x.strip() for x in y_pred]


with open('valid_new.json', 'r', encoding='utf-8') as fp:
    test_data = json.load(fp)
    test_data = [x['langid'] for x in test_data]

# calculate accuracy
accuracy = accuracy_score(test_data, y_pred)
print("Accuracy: ", accuracy)