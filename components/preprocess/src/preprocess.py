#!/usr/bin/env python3
import argparse
import os
import joblib
import logging
import pickle
from ast import literal_eval

from pathlib import Path
import pandas as pd

# disable SettingWithCopyWarning in Pandas
pd.options.mode.chained_assignment = None  # default='warn'


PREPROCESS_FILE = 'processor_state.pkl'


# Function to read data
def read_data(input1_path):
    with open(input1_path, mode='r') as input1_file:
        print('processing')
        print('input file', input1_file)
        csv_data = pd.read_csv(input1_file, error_bad_lines=False)
        return csv_data
        

# Function doing the actual work



# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='My program description')
parser.add_argument('--input1-path', type=str, help='Path of the local file containing the Input 1 data.') # Paths should be passed in, not hardcoded


parser.add_argument('--output-x-path', type=str, help='')
#parser.add_argument('--output-x-path-file', type=str, help='Path of the local file where the Output x-path-file data should be written.') # Paths should be passed in, not hardcoded'

parser.add_argument('--output-y-path', type=str, help='')
#parser.add_argument('--output-y-path-file', type=str, help='Path of the local file where the Output y-path-file data should be written.') # Paths should be passed in, not hardcoded'

parser.add_argument('--output-preprocessing-state-path', type=str, help='')
#parser.add_argument('--output-preprocessing-state-path-file', type=str, help='')

args = parser.parse_args()


# read data
data = read_data(args.input1_path)

# prints information about the data type and number of missing values
data.info()

# view some basic statistical details like percentile, mean, std etc. of numeric values
data.describe().T

# remove not required columns
data = data.drop('original_title', axis = 1, inplace = True)

# print the first 5 rows
print(data.head())

# Handling the Json Columns
# Applying the literal_eval function of ast on all the json columns
json_cols = ['cast','crew','genres','keywords','production_companies','production_countries','spoken_languages']
for col in json_cols:
    data[col] = data[col].apply(literal_eval)
    

# Extracting the features from Json Columns
# 1. list of Genres names (from Genres column)
# 2. Jobs of the Crew members (from Crew column)
# 3. Percentage of voice artists among total cast (from cast column)


# Helper Functions for the same
# function to get the names of the movies genre
def get_genre(x):
    if(isinstance(x, list)):
        genre = [i['name'] for i in x]
    
    return genre

# function to get the jobs of the crew members    
def get_jobs(x):
    if(isinstance(x, list)):
        jobs = [i['job'] for i in x]
    return jobs

# function to get the target/label (Animation == 1 / Not_Animation == 0)    
def get_labels(x):
    if(len(x)==0):
        return np.nan
    elif('Animation' in x):
        return 1
    else:
        return 0

# Get percentage of voice artists among total cast
def get_characternames(x):
    if(isinstance(x, list)):
        chr_name = [i['character'] for i in x]
        countc = 0
        for j in chr_name:
            if('(voice)' in j):
                countc += 1
        if(len(chr_name)!=0):
            return (countc/len(chr_name))
        else:
            return 0

# function to get crew memebers whose jobs are Costume Design
def get_costume_labels(x):
    if 'Costume Design' in x:
        return 1
    else:
        return 0
        
# function to get the genre department with the Lighting role
def get_genre_cd(x):
    if(isinstance(x, list)):
        dept = [i['department'] for i in x]
    if 'Lighting' in dept:
        return 0
    else:
        return 1

# Applying the above functions 
data['genres'] = data['genres'].apply(get_genre)
data['crew_jobs'] = data['crew'].apply(get_jobs)
data['percent_of_voice_artists'] = data['cast'].apply(get_characternames)
data['labels'] = data['genres'].apply(get_labels)


# Rounding off the percentage to 3 decimal places
for x in range(0,len(data['percent_of_voice_artists'])):
    data['percent_of_voice_artists'][x] = np.round(data['percent_of_voice_artists'][x],3)


# number of Labels missing / Null values  
data.labels.isna().sum()


# dealing with Labels missing values
idxsc = data[((data.labels != 1) & (data.labels != 0))].index
data.drop(idxsc, inplace = True)
data.reset_index(drop= True, inplace= True)

# checking for dataset Features with missing values
data.isna().sum()

# check the number of animated and non_animated movies
AnimatedMoviesCount = np.sum(data['labels'] == 1)
NotAnimatedMoviesCount = np.sum(data['labels'] == 0)

print("Number of Animated Movies are: ", AnimatedMoviesCount)
print("Number of Not Animated Movies are: ", NotAnimatedMoviesCount)


# Apply the get_costume_labels function
data['costume'] = data['crew_jobs'].apply(get_costume_labels)

data.costume.value_counts()


# Apply get_genre_cd function
data['lighting_dept'] = data['crew'].apply(get_genre_cd)

data.lighting_dept.value_counts()


# analysis to get the Average Budget of Animated Movie
c = np.where(data.labels==1)[0]
sum_budget = 0
for x in c:
    sum_budget += data.budget[x]
avg_budget = sum_budget/len(c)
print("Average Budget of Animated Movie: ",str(avg_budget))


# Taking into account only those movies having atleast 7 crew members
# So as to handle the quality of training data Tested for multiple values, but 7 yielded best result
idx=[]
for x in range(0,data.shape[0]):
    if len(data.crew_jobs[x])>7:
        idx.append(x)
print("Number of Movies with more than 7 crew members: ",str(len(idx)))

df = data.iloc[idx,:]


# Get the number of animated and non_animated movies
AnimatedMoviesCount2 = np.sum(df['labels'] == 1)
NotAnimatedMoviesCount2 = np.sum(df['labels'] == 0)

print("Number of Animated Movies are: ", AnimatedMoviesCount2)
print("Number of Not Animated Movies are: ", NotAnimatedMoviesCount2)


# Converting 'crew_jobs' from list to string (in lower form) via join function
def join_strings(x):
    return ", ".join(x)

def str_lower(x):
    return x.lower()

df['crew_jobs'] = df['crew_jobs'].apply(join_strings)
df['crew_jobs'] = df['crew_jobs'].apply(str_lower)


# get the number of labels
df['labels'].value_counts() 

# Get the features and labels
X = df['crew_jobs']
y = df['labels']



# export features and labels for training
with open(args.output_x_path, 'w') as output_X:
    pickle.dump(X, output_X)
    
    
with open(args.output_y_path, 'w') as output_y:
    pickle.dump(y, output_y)


# export preprocessing state, required for custom prediction route used
# during inference
preprocess_output = args.output_preprocessing_state_path + '/' + PREPROCESS_FILE
with open(preprocess_output, 'w') as output_preprocessing_state:
    pickle.dump(processor, output_preprocessing_state)


 
# Creating the directory where the output file will be created (the directory may or may not exist).
# Path(args.output1_path).parent.mkdir(parents=True, exist_ok=True)

# writing x and y path to a file for downstream tasks
Path(args.output_x_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_x_path_file).write_text(args.output_x_path)

Path(args.output_y_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_y_path_file).write_text(args.output_y_path)

Path(args.output_preprocessing_state_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_preprocessing_state_path_file).write_text(args.output_preprocessing_state_path + '/' + PREPROCESS_FILE)






















