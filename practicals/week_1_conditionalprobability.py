# -*- coding: utf-8 -*-
"""
##summarize_dataset, summarize_by_class, calculate_class_probabilities NAKIJKEN 
LET OP ALS class_value, dit is nu 'target'

Verdeling training en test maken. 
"""
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
import copy
import matplotlib.pyplot as plt

# Split the dataset by classifier target, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		valuetarget = dataset[i]
		target = dataset[i,1]
		if (target not in separated):
			separated[target] = list()
		separated[target].append(valuetarget)
	return separated

# Calculate the mean, std and count for each column in the dataset
def summarize_dataset(dataset):
	summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
 
# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

# Gaussian probability distribution function for x
def gaussian_probability(x, mean, std):
    gaussian=(1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x-mean)**2 / (2 * std**2 )))
    return gaussian
 
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, std, _ = class_summaries[i]
			probabilities[class_value] *= gaussian_probability(row[i], mean, std)
	return probabilities
 
def normalize(probs):
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]




breast_cancer = load_breast_cancer()
feature_names = breast_cancer.feature_names
#verdeling nog aanpassen met training en test set, nu resp. 300 : 269
probability_0 = np.ones((269,30))
probability_1 = np.ones((269,30))
for k in range(0,len(feature_names)):
    X_train = breast_cancer.data[:300, k]
    y_train = breast_cancer.target[:300]
    X_test = breast_cancer.data[300:, k]
    y_test = breast_cancer.target[300:]

#onderstaande regel wordt volgens mij verder nergens gebruikt in het script
    dataset_0 = np.append(np.transpose(X_train), np.transpose(y_train))

#Create datasets which contains the values for the k-feature and the corresponding target
    dataset = np.ones((len(X_train),2))
    dataset_test = np.ones((len(X_test),2))
    for i in range(0,len(X_train)):
        dataset[i] = [X_train[i],y_train[i]]
        for i in range(0,len(X_test)):
            dataset_test[i] = [X_test[i],y_test[i]]
    
    
    summaries = summarize_by_class(dataset)


    for j in range(0,len(y_test)):
        probability = calculate_class_probabilities(summaries, dataset_test[j])
        prob_0 = probability[0]
        prob_1 = probability[1]
        probs = [prob_0,prob_1]
        prob_0,prob_1 = normalize(probs)
        
        probability_0[j,k] = prob_0
        probability_1[j,k] = prob_1
    
fig, axs = plt.subplots(6,5, figsize=(45, 45), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for i in range(30):

    axs[i].plot(probability_0[:,i], 'bx', label='malignant')
    axs[i].plot(probability_1[:,i], 'rx', label='benign')
    
    axs[i].set_title(feature_names[i])
    axs[i].set_ylabel('Probability')
   # axs[i].legend()