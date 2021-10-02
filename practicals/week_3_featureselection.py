import numpy as np
import pandas as pd
pwd="D:\8dm50-machine-learning\Machine_Learning\practicals"
gene_expression = pd.read_csv(pwd + "./data/RNA_expression_curated.csv", sep=',', header=0, index_col=0)
drug_response = pd.read_csv(pwd + "./data/drug_response_curated.csv", sep=',', header=0, index_col=0)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

split = round(len(gene_expression)*0.7)
X_train = gene_expression.iloc[:split,:]
y_train = drug_response.iloc[:split,:]
X_test = gene_expression.iloc[split:,:]
y_test = drug_response.iloc[split:,:]

def lasso_regression(X_train_data,y_train_data,X_test_data,y_test_data,alpha_list = []):
    scaler = StandardScaler()

    model = Pipeline([
            ("scaler", scaler),
            ("lasso", Lasso())
            ])

    #     print(model.best_params_)
    clf = GridSearchCV(estimator=model,param_grid = {'lasso__alpha':alpha_list},cv=4)
    clf.fit(X_train_data,y_train_data)
    y_predicted = clf.predict(X_test_data)
    best_alpha=clf.best_params_
    #clf.best_estimator_
    
    return clf, y_predicted, best_alpha


def plot_score(clft):
    fig = plt.figure()
    results = clft.cv_results_['mean_test_score']
    plt.plot(alpha_list,results)

    plt.plot(alpha_list[int(np.where(results == np.amax(results))[0])],np.amax(results),'ro')
    plt.xlabel('alpha ')
    plt.ylabel('mean test score')

    print(clft.cv_results_['rank_test_score'])
    
alpha_list = np.linspace(0,1,11)
clf, y_predicted, best_alpha = lasso_regression(X_train,y_train,X_test,y_test,alpha_list)
plot_score(clf)

### STUK CODE BEGIN VANAF HIER 

featurenames=list(gene_expression.columns)
coefficients = clf.best_estimator_.named_steps['lasso'].coef_
coefficients = np.abs(coefficients) #take absolute value
resultingfeatures = np.array(featurenames)[importance > 0] #select feature names that aren't set to 0 by lasso