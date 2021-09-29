## Python functions of exercises week 2

#%% import
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc


#%% Exercise Polynominal regression

def polynominal_regression(X_train_data,y_train_data,X_test_data,y_test_data,polynominal_list = []):
    linearregression = linear_model.LinearRegression()
    '''Function that performs a grid search on a specific polynomial model and calculates the predicted value for a the test data. '''
    scaler = StandardScaler()
    model = Pipeline([
        ("scaler", scaler),
        ("poly", PolynomialFeatures()),
        ("linear regression",linearregression)
        ])
    
    #     print(model.best_params_)
    clf = GridSearchCV(estimator=model,param_grid = {'poly__degree':polynominal_list},cv=4)
    clf.fit(X_train_data,y_train_data)
    y_predicted = clf.predict(X_test_data)
    return clf, y_predicted



def plot_score(clft, polynominal_list = []):
    ''' function that plots and prints the results from the grid search.'''
    fig = plt.figure()
    results = clft.cv_results_['mean_test_score']
    plt.plot(polynominal_list,results)

    plt.plot(polynominal_list[int(np.where(results == np.amax(results))[0])],np.amax(results),'ro')
    plt.xlabel(' polynomial order ')
    plt.ylabel('mean test score')

    print(clft.cv_results_['rank_test_score'])
    
#%% Exercise ROC curve analysis
    
def KNN_model_ROC_plots(X_train, y_train, X_test, y_test, k_list):
    """
    This function runs a knn model for a list of different k-values.
    Next the predictions will be determined, ROC curve is determined and
    the area under the curve (AUC) is calculated. Finally the ROC curves
    will be plotted for all different values for k."""
    for k in k_list:
        scaler = StandardScaler()
        # initialize a k-NN classifier
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        # Create the pipeline
        model = Pipeline([
                      ("scaler", scaler),
                      ("knn", knn)
                     ])
        # train the model using the training dataset
        model.fit(X_train, y_train)
        # make predictions using the testing dataset
        prediction = model.predict_proba(X_test)
        # calculate the Receiver Operating Characteristics (True positive rate, false positive rate and treshold)
        fpr, tpr, threshold = roc_curve(y_test, prediction[:,1])
        # calculate area under curve
        roc_auc = auc(fpr, tpr)
        
    
        # plot the ROC curves
        plt.plot(fpr, tpr, label = 'k = %d, AUC = %0.2f' % (k,roc_auc))
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('Receiver Operating Characteristic')
    
    plt.show()