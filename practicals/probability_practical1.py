import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
import copy
import matplotlib.pyplot as plt

#places the classes as keys in a dictionary, with an array as value containing:mean,std,length
def separate_classes(dataset):
    dct = {}
    for i in dataset:
        target = i[1]
        value = i[0]
        if(target not in dct):
            dct[target] = list()
            dct[target].append(value)  
        else:
            dct[target].append(value)
    for i in dct.keys():
        dct[i] = [(np.mean(dct[i]),np.std(dct[i]),len(dct[i]))]
    return dct


# Gaussian probability distribution function for x
def gauss(x, mean, std):
    gaussian=(1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x-mean)**2 / (2 * std**2 )))
    return gaussian
 
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(data_sum, row):
    rows = sum([data_sum[label][0][2] for label in data_sum]) #heel abstract
    probabilities = dict()
    for c_value, c_data in data_sum.items():
        probabilities[c_value] = data_sum[c_value][0][2]/float(rows)
        for i in range(len(c_data)):
            mean, std, _ = c_data[i]
            probabilities[c_value] *= gauss(row[i], mean, std)
    return probabilities
 
def normalize(prob):
    prob_factor = 1 / sum(prob)
    return [prob_factor * p for p in prob]



#start
#load data
breast_cancer = load_breast_cancer()
feature_names = breast_cancer.feature_names
test_num = 169
train_num = 400

y_train = breast_cancer.target[:train_num] #zou niet in for loop hoeven, is wel netter
y_test = breast_cancer.target[train_num:]  #zou niet in for loop hoeven, is wel netter
print(y_test)

#create matrix for probability
probability_zero = np.ones((test_num,30))
probability_one = np.ones((test_num,30))

#go through every feature
for k in range(0,len(feature_names)):
    X_train = breast_cancer.data[:train_num, k]
    X_test = breast_cancer.data[train_num:, k]

    #Create datasets which contains the values for the k-feature and the corresponding target
    dataset = np.ones((len(X_train),2))
    dataset_test = np.ones((len(X_test),2))
    
    for i in range(0,len(X_train)):
        dataset[i] = [X_train[i],y_train[i]]
    for q in range(0,len(X_test)):
        dataset_test[q] = [X_test[q],y_test[q]]
    
    #create dict with classes as keys, and values of mean,std,length
    data_sum = separate_classes(dataset)

    for j in range(0,len(y_test)):
        probability = calculate_class_probabilities(data_sum, dataset_test[j])
        prob_zero = probability[0]
        prob_one = probability[1]
        probs = [prob_zero,prob_one]
        prob_zero,prob_one = normalize(probs)
        
        probability_zero[j,k] = prob_zero
        probability_one[j,k] = prob_one

fig, axs = plt.subplots(6,5, figsize=(45, 45), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for i in range(30):

    axs[i].plot(probability_zero[:,i], 'bx', label='malignant')
    axs[i].plot(probability_one[:,i], 'rx', label='benign')
    
    axs[i].set_title(feature_names[i])
    axs[i].set_ylabel('Probability')
