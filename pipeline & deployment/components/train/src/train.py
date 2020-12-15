import argparse
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from tensorflow import gfile
from tensorflow.python.lib.io import file_io
from keras.models import Model, Input
from keras.callbacks import TensorBoard
import spacy
from spacy.lang.en import STOP_WORDS
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


MODEL_FILE = 'movie_success_model.pkl'


def load_feature(input_x_path):
  with open(input_x_path, 'rb') as input_x_file:
    return pickle.load(input_x_file)


def load_label(input_y_path):
  with open(input_y_path, 'rb') as input_y_file:
    return pickle.load(input_y_file)
    

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='My model training program description')
parser.add_argument('--input-x-path', type=str, help='')
parser.add_argument('--input-y-path', type=str, help='')
parser.add_argument('--input-job-dir', type=str, help='')

parser.add_argument('--output-model-path', type=str, help='')
parser.add_argument('--output-model-path-file', type=str, help='')

args = parser.parse_args()

print(os.path.dirname(args.output_model_path))

print(args.input_x_path)
print(args.input_y_path)
print(args.input_job_dir)
print(args.output_model_path)
print(args.output_model_path_file)


X = load_feature(args.input_x_path)
y = load_label(args.input_y_path)


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=53)


# initialize tensorboard
tensorboard = TensorBoard(
    log_dir=os.path.join(args.input_job_dir, 'logs'),
    histogram_freq=0,
    write_graph=True,
    embeddings_freq=0)

callbacks = [tensorboard]


# function to output our scores 
def score_output(y_test, y_pred):
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print('The Accuracy on The Test Set is: %s' % accuracy)
    

# model
nlp = spacy.load('en_core_web_sm')

# instantiate stopwords to use
stop_words_str = " ".join(STOP_WORDS)
stop_words_lemma = set(word.lemma_ for word in nlp(stop_words_str))

additional_words = ['editor', 'director', 'producer', 'writer', 'assistant', 'sound']

for word in additional_words:
    stop_words_lemma = stop_words_lemma.union({word})
    

# define the lemmatizer function
def lemmatizer(text):
    return [word.lemma_ for word in nlp(text)]
     

# Without Stop Words
bow = TfidfVectorizer(ngram_range = (1,1))

model = Pipeline([('bag_of_words', bow),('classifier', SVC())])
model.fit(X_train,y_train)

print("Without Stop Words")
print('Training accuracy: {}'.format(model.score(X_train,y_train)))
y_pred = model.predict(X_test)
score_output(y_test, y_pred)


# save model
print('saved model to ', args.output_model_path)  
#with open(MODEL_FILE, mode='rb') as input_f:
 #   with open(args.output_model_path + '/' + MODEL_FILE, mode='wb+') as output_f:
  #      pickle.dump(model, file)


model.save(MODEL_FILE)
with file_io.FileIO(MODEL_FILE, mode='rb') as input_f:
  with file_io.FileIO(args.output_model_path + '/' + MODEL_FILE, mode='wb+') as output_f:
    output_f.write(input_f.read())
    
    
#with open(args.input1_path, 'r') as input1_file:
 #   with open(args.output1_path, 'w') as output1_file:
  #      do_work(input1_file, output1_file, args.param1)


#with open('mypickle.pickle', 'wb') as f:
 #   pickle.dump(some_obj, f)


# write out metrics
accuracy = model.score(X_train,y_train)

metrics = {
    'metrics': [{
        'name': 'accuracy-score',
        'numberValue': accuracy,
        'format': "PERCENTAGE",
    }]
}   

#with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
#  json.dump(metrics, f)
  
with open('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)


# write out TensorBoard viewer
metadata = {
    'outputs': [{
        'type': 'tensorboard',
        'source': args.input_job_dir,
    }]
}

with open('/mlpipeline-ui-metadata.json', 'w') as f:
  json.dump(metadata, f)
  

Path(args.output_model_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_model_path_file).write_text(args.output_model_path)




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    