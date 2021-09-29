## Python functions of exercises week 2

def polynominal_regression(X_train_data,y_train_data,X_test_data,y_test_data,polynominal_list = []):
    linearregression = linear_model.LinearRegression()
    # train the model using the training dataset
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



def plot_score(clft):
    fig = plt.figure()
    results = clft.cv_results_['mean_test_score']
    plt.plot(polynominal_list,results)

    plt.plot(polynominal_list[int(np.where(results == np.amax(results))[0])],np.amax(results),'ro')
    plt.xlabel(' polynomial order ')
    plt.ylabel('mean test score')

    print(clft.cv_results_['rank_test_score'])
    
    
def KNN_model_ROC_plots(X_train, y_train, X_test, y_test, k_list):
    for k in k_list:
    
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