import pandas as pd

import numpy as np



def preprocess_classification_dataset():

    

    train_df = pd.DataFrame(pd.read_csv('train.csv'))

    

    train_feat_df = train_df.iloc[:,:-1] # grab all columns except the last one

    train_output = train_df[['output']]

    

    X_train = train_feat_df.values

    y_train = train_output.values

    

    val_df = pd.DataFrame(pd.read_csv('val.csv'))

    

    val_feat_df = val_df.iloc[:, :-1]

    val_output = val_df[['output']]

    

    x_val = val_feat_df.values

    y_val = val_output.values

    

    test_df = pd.DataFrame(pd.read_csv('test.csv'))

    

    test_feat_df = test_df.iloc[:, :-1]

    test_output = test_df[['output']]

    

    x_test = test_feat_df.values

    y_test = test_output.values





    

    return X_train, y_train, x_val, y_val, x_test, y_test



def knn_classification(x_train, y_train, x_new, k=5):

    

    distance = []

    

    for i in range(len(x_train)):

        points = []

        for j in range(len(x_train[i])):

            points.append((x_train[i][j]-x_new[j])**2)

        

        distance.append(sum(points)**.5)

        

    Min = np.argsort(distance)

        

    y_val = []

    for i in range(k):

        y_val.append(y_train[Min[i]])

        

    values, counts = np.unique(y_val, return_counts=True)

    

    if len(counts) > 1:

        if counts[0] == counts[1]:

            y_new = 1

        else:

            y_new = values[np.argmax(counts)]

    else:

        y_new = values[np.argmax(counts)]

    

    return y_new



def logistic_regression_training(x_train, y_train, alpha=0.01, max_iters=5000, random_seed=1):

    

    x_train_new = np.hstack((np.ones((len(x_train), 1)), x_train)) 

    np.random.seed(random_seed) # for reproducibility

    weights = np.random.normal(loc=0.0, scale=1.0, size=(len(x_train_new[0]), 1))

    

    

    

    for i in range(max_iters):

        Xw = sigmoid(np.matrix(x_train_new)@weights)

        Xw_Y = Xw - np.matrix(y_train)

        weights = weights - ((alpha*np.matrix.transpose(x_train_new))@Xw_Y)

    

    return weights



def  logistic_regression_prediction(X, weights, threshold=0.5):

    x_new = np.hstack((np.ones((len(X), 1)), X))

    y_pred = []

    prob = sigmoid(np.matrix(x_new)@np.matrix(weights))

    

    for y in prob:

        if y > threshold:

            y_pred.append(1)

        else:

            y_pred.append(0)

    

    

    return np.array(y_pred)



def sigmoid(x):

    return 1/(1+np.matrix(np.exp(-x)))



def model_selection_and_evaluation(alpha=0.01, max_iters=5000, random_seed=1, threshold=0.5):

    

    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_classification_dataset()

    

    Accuracy = []

    methods = ['1nn', '3nn', '5nn', 'logistic regression']

    y_val = np.array(y_val)

    

    nn1_pred = knn_evaluation(x_train, y_train, x_val, 1)

    nn1_pred = np.array(nn1_pred)

    

    Accuracy.append((y_val.flatten() == nn1_pred.flatten()).sum()/y_val.shape[0])

    

    nn3_pred = knn_evaluation(x_train, y_train, x_val, 3)

    nn3_pred = np.array(nn3_pred)

    

    Accuracy.append((y_val.flatten() == nn3_pred.flatten()).sum()/y_val.shape[0])

    

    nn5_pred = knn_evaluation(x_train, y_train, x_val, 5)

    nn5_pred = np.array(nn5_pred)

    

    Accuracy.append((y_val.flatten() == nn5_pred.flatten()).sum()/y_val.shape[0])

    

    weights = logistic_regression_training(x_train, y_train, alpha, max_iters, random_seed)

    

    logistic_pred = logistic_regression_prediction(x_val, weights, threshold)

    logistic_pred = np.array(logistic_pred)

    

    Accuracy.append((y_val.flatten() == logistic_pred.flatten()).sum()/y_val.shape[0])

    

    if Accuracy.index(max(Accuracy)) == 0:

        X_train_val_merge = np.vstack([x_train, x_val]) 

        y_train_val_merge = np.vstack([y_train, y_val])

        

        nn1b_pred = knn_evaluation(X_train_val_merge, y_train_val_merge, x_test, 1)

        nn1b_pred = np.array(nn1b_pred)

        

        test_accuracy = (y_test.flatten() == nn1b_pred.flatten()).sum()/y_test.shape[0]

    

    elif Accuracy.index(max(Accuracy)) == 1:

        X_train_val_merge = np.vstack([x_train, x_val]) 

        y_train_val_merge = np.vstack([y_train, y_val])

        

        nn3b_pred = knn_evaluation(X_train_val_merge, y_train_val_merge, x_test, 3)

        nn3b_pred = np.array(nn3b_pred)

        

        test_accuracy = (y_test.flatten() == nn3b_pred.flatten()).sum()/y_test.shape[0]

        

    elif Accuracy.index(max(Accuracy)) == 2:

        X_train_val_merge = np.vstack([x_train, x_val]) 

        y_train_val_merge = np.vstack([y_train, y_val])

        

        nn5b_pred = knn_evaluation(X_train_val_merge, y_train_val_merge, x_test, 5)

        nn5b_pred = np.array(nn5b_pred)

        

        test_accuracy = (y_test.flatten() == nn5b_pred.flatten()).sum()/y_test.shape[0]

    

    elif Accuracy.index(max(Accuracy)) == 3:

        X_train_val_merge = np.vstack([x_train, x_val]) 

        y_train_val_merge = np.vstack([y_train, y_val])

        

        weightsb = logistic_regression_training(X_train_val_merge, y_train_val_merge, alpha, max_iters, random_seed)

        

        logisticb_pred = logistic_regression_prediction(x_test, weightsb, threshold)

        logisticb_pred = np.array(logisticb_pred)

        

        test_accuracy = (y_test.flatten() == logisticb_pred.flatten()).sum()/y_test.shape[0]



    

    return methods[Accuracy.index(max(Accuracy))], Accuracy, test_accuracy



def knn_evaluation(x_train, y_train, x_new, k):

    y_pred = []

    

    for x in x_new:

        y_pred.append(knn_classification(x_train, y_train, x, k))

        

    return y_pred

if __name__ == "__main__":

    

    a, b, c, d, e, f= preprocess_classification_dataset()

    x_new = a[0]

    Dis = knn_classification(a, b, x_new)

    Dir = logistic_regression_training(a, b)

    accuracy, test, best = model_selection_and_evaluation(alpha=0.0001, max_iters=10, random_seed=321, threshold=0.7)

    

    y = [1,1,1,2,2,1]

    x = [3,2,3,4,3,1]

    p1 = [[0,1],[3,2],[1,2]]

    p1 = np.array(p1)

    y = np.array(y)

    w = [[3],[3]]

    p2 = [[2,1],[1,2]]

    xnew = 3

    ax = [[3,3,3],[3,3,3]]

    d = []

    for v in x:

        d.append(((xnew-v)**2)**(.5))

    print (np.unique(d, return_counts=True))

    print(accuracy, test, best)

    print(np.hstack((np.ones((len(ax),1)),ax)))