
# coding: utf-8

# In[2]:

import numpy as np
from matplotlib import pyplot as plt
import bz2
from sklearn.datasets import svmlight_format
from scipy.sparse import csr_matrix
from sklearn import svm, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE,RFECV, SelectKBest
from sklearn.svm import SVC,LinearSVC
import math
from sklearn.grid_search import GridSearchCV
def plot_data(x, Golub, RFE, ML1, flag) :
    fig = plt.figure(figsize=(500,500))
    
    ax = plt.subplot(111)
    ax.plot(x, Golub, label = 'Using Golub Score As Selector')
    
    ax.plot(x, RFE, label = 'Using L2 RFE As Selector')
    ax.plot(x, ML1, label = 'Using Modified L1 As Selector')

    
    ax.legend(bbox_to_anchor=(1.1, 1.05)) 
#     ax.plot(title = "Awesome")
    plt.xlabel('Number Of Features (Log2)')
    plt.ylabel('Accuracy')
    plt.title('For Dataset: ' + flag)
    plt.show()
def golub_score(X, y):
    X_Positive = []
    X_Negative = []
    y_Positive = []
    y_Negative = []

    for i in range(len(y)):
        if(y[i] == -1):
            y_Negative.append(y[i])
            X_Negative.append(X[i])
        elif(y[i] == 1):
            y_Positive.append(y[i])
            X_Positive.append(X[i])

    X_Positive = np.asarray(X_Positive)
    X_Negative = np.asarray(X_Negative)
    y_Positive = np.asarray(y_Positive)
    y_Negative = np.asarray(y_Negative)

#     print "Positive Datasize",X_Positive.shape
#     print "Negative Datasize",X_Negative.shape
#     print "Positive Labels",y_Positive.shape
#     print "Positive Labels",y_Negative.shape

    Average_Positive_Example = np.average(X_Positive, axis = 0)
    Average_Negative_Example = np.average(X_Negative, axis = 0)

    Standard_Positive_Example = np.std(X_Positive, axis=0)
    Standard_Negative_Example = np.std(X_Negative, axis=0)

#     print "Average Positive", Average_Positive_Example.shape
#     print "Average Negative", Average_Negative_Example.shape
#     print "Standard Positive", Standard_Positive_Example.shape
#     print "Standard Negative", Standard_Negative_Example.shape

    Average_Difference = np.absolute(Average_Positive_Example - Average_Negative_Example)
    Standard_Sum_Temporary = Standard_Positive_Example + Standard_Negative_Example

    Standard_Sum = []
    for x in Standard_Sum_Temporary:
        if(x == 0):
            x = 1
        Standard_Sum.append(x)

    Golub = Average_Difference / Standard_Sum
    Golub = np.asarray(Golub)
    return Golub, Golub

def compute_accuracy_of_selector(custom_selector, selector_name, classifier_name, number_of_feature, X, y):
    
    
    print "Classifier", classifier_name
    k = number_of_feature
    print "Number Of Feature", k
    print "Selector Name", selector_name
    classifier = LinearSVC(penalty=classifier_name, dual=False)
    cross_valid = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    
    if selector_name == "Golub":
        select = SelectKBest(custom_selector, k)
        pipeline = make_pipeline(select, classifier)
        gotAccuracy = np.mean(cross_validation.cross_val_score(pipeline, X, y, cv=cross_valid, scoring='accuracy'))
        print "Accuracy Using Golub", gotAccuracy
        return gotAccuracy

def generate_data(flag):
    if (flag == 1):
        print "For Arcene Dataset"
        
        y_trainData=np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.labels")
        print "Arcene Training Dataset Label: ", y_trainData.shape
        X_trainData=np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.data")
        print "Arcene Training Dataset Size: ", X_trainData.shape

        y_validData=np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/arcene_valid.labels")
        print "Arcene Valid Dataset Label: ", y_validData.shape
        X_validData=np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_valid.data")
        print "Arcene Valid Dataset Size: ", X_validData.shape

        X = np.concatenate((X_trainData, X_validData), axis=0)
        y = np.concatenate((y_trainData, y_validData), axis=0)

        print "After Concate: X Size: ", X.shape
        print "After Concate: y Size: ", y.shape
    
    else:
        print "For Leukemia Dataset"
        
        data_train = svmlight_format.load_svmlight_file('leu.bz2')
        data_test = svmlight_format.load_svmlight_file('leu.t.bz2')
        
        X_train = data_train[0].toarray()
        y_train = data_train[1]

        X_test = data_test[0].toarray()
        y_test = data_test[1]

        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        print "After Concate: X Size: ", X.shape
        print "After Concate: y Size: ", y.shape

    return X, y

def rfe_feature_select(X, y, dataset):
    k = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
#     k = []
    if(dataset == 1):
        print "Arcene Dataset RFE Selection"
        k.append(8192)
    else:
        print "Leukemia Dataset RFE Selection"
        
    cross_valid = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    accuracys = []
    for k_value in k:
        print "K", k_value
        classifier = LinearSVC(penalty="l2", dual = False)
        selector = RFE(classifier, step=0.1, n_features_to_select=k_value)
        pipeline = make_pipeline(selector, classifier)
        got_accuracy = np.mean(cross_validation.cross_val_score(pipeline, X, y, cv=cross_valid, scoring='accuracy'))
        print "Accuracy", got_accuracy
        accuracys.append(got_accuracy)
    return k, accuracys
def generateKSubSample(X, y):
#     print "In Generate", X.shape[0]
    k = 5
    if X.shape[0] >= 100:
        k = 8
    else:
        k = 2
#     print "Part Will Be", k
    number_rows = X.shape[0]
    
    
#     number_Train = int(round(number_rows*0.8))
#     number_Test = number_rows - number_Train
# #     print number_Train,number_Test,number_Train+number_Test
    
    rows = np.arange(number_rows)
#     np.random.shuffle(rows)
# #     print rows
# #     print rows
    
#     training = rows[:number_Train]
#     testing = rows[number_Train:]
# #     print "y", y
#     y = y.reshape((-1,1))
# #     print y
#     X_training = X[training,:]
#     y_training = y[training,:]
#     X_testing = X[testing,:]
#     y_testing = y[testing,:]
    
#     y_training = y_training.reshape(1, -1)[0]
# #     print y
# #     print y_training
# #     print "Random",len(X_training)
#     k = 1
    X_training = X
    y_training = y
#     k = 1
#     X_split_data =np.split(X_training, k)
# #     
#     y_split_data = np.split(y_training, k)
    y_training = y_training.reshape(-1, 1)
#     k= 8
    fold_examples = math.floor(number_rows / k)
#     print "Fold Entry", fold_examples
#     print "Total Rows", rows
    X_split_data = []
    y_split_data = []
    
    for i in range(k):
        select_rows = rows[i*fold_examples : (i+1)*fold_examples]
#         print "Range I", i, " Row", select_rows
        X_Selected = X_training[select_rows,:]
        y_Selected = y_training[select_rows,:]
        y_Selected = y_Selected.reshape(1, -1)[0]
        
        
        X_split_data.append(X_Selected)
        y_split_data.append(y_Selected)
#         print "Select X", X_Selected
#         print "Select Y", y_Selected
#         print "Selected Till", (i+1) * fold_examples
#     print "Final Start", (k*fold_examples)
#     print "Final Num", number_rows

#     print "Final", (fold_examples * k)
#     print "final to", (number_rows - 1)
    if(((fold_examples*k)+1)<=(number_rows-1)):
        final_row = rows[((fold_examples*k)+1):(number_rows-1)]
    #     print "Final", final_row
        X_Selected = X_training[final_row,:]
        y_Selected = y_training[final_row,:]
        y_Selected = y_Selected.reshape(1, -1)[0]
        X_split_data.append(X_Selected)
        y_split_data.append(y_Selected)
#     print "Before Split", X_training
#     print X_split_data[0].shape
    return X_split_data, y_split_data
def L1_SVM_Coef(X, y):
    run_svm = 1
    score_vector_fold = np.zeros(X.shape[1])
    for count_down in range((run_svm)):
        classifier = LinearSVC(penalty='l1', dual=False)

        classifier = classifier.fit(X, y)

        for i in range(len(classifier.coef_[0])):

            if(classifier.coef_[0][i] !=0):
                score_vector_fold[i] = 1

    return score_vector_fold

def L1_Modified_Features(X, y):
#     print "In Modified F", X.shape
    X_Split, y_Split = generateKSubSample(X, y)
#     print "After Generate", X_Split
    score_vector = np.zeros(X.shape[1])

    for i in range(len(X_Split)):
#         print "Fold: ", i
        score_vector = score_vector + L1_SVM_Coef(X_Split[i], y_Split[i])
    return score_vector, score_vector
    
def modified_L1_feature_select(X, y, dataset):
    k = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
#     k = []
    if(dataset == 1):
        print "Arcene Dataset Modified L1 Selection"
        k.append(8192)
    else:
        print "Leukemia Dataset Modified L1 Selection"
    cross_valid = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
#     X_Temp = X
#     y_Temp =y
    results = []
    for k_value in k:
        print "K", k_value
#         print "L1",L1_Modified_Features(X, y)
#         for i in L1_Modified_Features(X, y):
#             if i == 4:
#                 print "I", i
#         print "X Shape", X.shape[0]
#         print "In F Modified Select", X.shape
        select = SelectKBest(L1_Modified_Features, k_value)
        classifier = LinearSVC(penalty="l2", dual = False)
        pipeline = make_pipeline(select, classifier)
        gotAccuracy = np.mean(cross_validation.cross_val_score(pipeline, X, y, cv=cross_valid, scoring='accuracy'))
        print "Accuracy Using L1 Modified", gotAccuracy
        results.append(gotAccuracy)
    return k, results

def model_selection(X, y, flag):
    
    k = []
    if flag == 1:
        k = [512, 1024, 2048, 4096, 8192]
    else:
#         k = [1, 2, 4, 8, 16]
        k = [32]
    classifier = LinearSVC(penalty = 'l2', dual=False)
    cross_valid = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    best_global_accuracy = 0
    best_golub_accuracy = 0
    best_golub_answer = ""
    best_global_answer = ""
    Cs = np.logspace(-4, 3, 8)
#     print "C Values", Cs
    dict_Cs = {'SVMlinear2__C':Cs}
    
    k_golub= k
#     1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 
#     k_golub = 4096
    for features in k_golub:
        print "Running Golub For Features", features
        selector_golub = SelectKBest(golub_score, features)


        pipeline_information = [('golub',selector_golub),('SVMlinear2',classifier)]
        pipeline_golub = Pipeline(pipeline_information)

        grid_search_golub = GridSearchCV(estimator = pipeline_golub, param_grid=dict_Cs, cv=cross_valid)
        grid_search_golub.fit(X, y)

        got_accuracy = grid_search_golub.best_score_
        print got_accuracy
        print grid_search_golub.best_params_
#         print best_golub_accuracy
#         print best_global_accuracy
        
        if got_accuracy>=best_golub_accuracy:
            best_golub_answer = "Feature " + str(features) + " Parameter" + str(grid_search_golub.best_params_)
            best_golub_accuracy = got_accuracy
        if got_accuracy>=best_global_accuracy:
            best_global_answer = "Golub Feature " + str(features) + " Parameter" + str(grid_search_golub.best_params_)
            best_global_accuracy = got_accuracy
    print "Best Combination Global: ", best_global_answer
    print "Best Global Results: ", best_golub_answer
    
    best_RFE_accuracy = 0
    best_RFE_answer = ""
    k_rfe = k
    for features in k_rfe:
        print "Running RFE For Features", features
#         selector_golub = SelectKBest(golub_score, features)

        selector_RFE = RFE(classifier, step=0.1, n_features_to_select=features)


        pipeline_information = [('RFE',selector_RFE),('SVMlinear2',classifier)]
        pipeline_RFE = Pipeline(pipeline_information)

        grid_search_RFE = GridSearchCV(estimator = pipeline_RFE, param_grid=dict_Cs, cv=cross_valid)
        grid_search_RFE.fit(X, y)

        got_accuracy = grid_search_RFE.best_score_
        
        print got_accuracy
        print grid_search_golub.best_params_

#         print best_RFE_accuracy
#         print best_global_accuracy
        
        if got_accuracy>=best_RFE_accuracy:
            best_RFE_answer = "Feature " + str(features) + " Parameter" + str(grid_search_golub.best_params_)
            best_RFE_accuracy = got_accuracy
        if got_accuracy>=best_global_accuracy:
            best_global_answer = "RFE Feature " + str(features) + " Parameter" + str(grid_search_golub.best_params_)
            best_global_accuracy = got_accuracy
            
    print "Best Combination Global: ", best_global_answer
    print "Best RFE Results: ", best_RFE_answer
    
    best_L1M_accuracy = 0
    best_L1M_answer = ""
    k_l1m = k
    for features in k_l1m:
        print "Running L1 Modified For Features", features
#         selector_golub = SelectKBest(golub_score, features)

        selector_LM = SelectKBest(L1_Modified_Features, features)


        pipeline_information = [('modified',selector_LM),('SVMlinear2',classifier)]
        pipeline_LM = Pipeline(pipeline_information)

        grid_search_LM = GridSearchCV(estimator = pipeline_LM, param_grid=dict_Cs, cv=cross_valid)
        grid_search_LM.fit(X, y)

        got_accuracy = grid_search_LM.best_score_
        
        print got_accuracy
        print grid_search_golub.best_params_

#         print best_L1M_accuracy
#         print best_global_accuracy
        
        if got_accuracy>=best_L1M_accuracy:
            best_L1M_answer = "Feature " + str(features) + " Parameter" + str(grid_search_golub.best_params_)
            best_L1M_accuracy = got_accuracy
        if got_accuracy>=best_global_accuracy:
            best_global_answer = "L1 Modified Feature " + str(features) + " Parameter" + str(grid_search_golub.best_params_)
            best_global_accuracy = got_accuracy
            
    print "Best Combination Global: ", best_global_answer
    print "Best L1 Modified Results: ", best_L1M_answer


def golub_feature_select(X, y, dataset):
    k = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
#     k = []
    if(dataset == 1):
        print "Arcene Dataset Golub Selection"
        k.append(8192)
    else:
        print "Leukemia Dataset Golub Selection"
    results = []
    for k_value in k:
        print "K", k_value
        accuracy_by_golub = compute_accuracy_of_selector(golub_score, "Golub", "l2", k_value, X, y)
        results.append(accuracy_by_golub)
    return k, results

if __name__=='__main__' :
#     X_Arcene,y_Arcene = generate_data(1)
#     model_selection(X_Arcene, y_Arcene, 1)

    X_L, y_L = generate_data(2)
    model_selection(X_L, y_L, 2)
    print "Done"
#     X = X_Arcene
#     y = y_Arcene
#     model_selection(X_Arcene, y_Arcene, 1)

#     golub_rank_arcene = golub_score(X_Arcene, y_Arcene)
#     print golub_rank_arcene
#     kValuesArceneGolub, accuracyValuesOfGolubArcene = golub_feature_select(X_Arcene, y_Arcene, 1)
# #     print kValues
# #     print accuracyValuesOfGolub
#     kValuesArceneRFE, accuracyValuesOfRFEArcene = rfe_feature_select(X_Arcene, y_Arcene, 1)
# #     print kValuesArceneRFE
# #     print accuracyValuesOfRFEArcene
#     kValuesArceneModified, accuracyValuesOfModifiedArcene = modified_L1_feature_select(X_Arcene, y_Arcene, 1)
# #     print kValuesArceneModified
# #     print accuracyValuesOfModifiedArcene
#     logOfK = np.log2(kValuesArceneRFE)
    
# #     print logOfK
# #     print accuracyValuesOfGolub
# #     print accuracyValuesOfGolubArcene
# #     print accuracyValuesOfRFEArcene
# #     print accuracyValuesOfModifiedArcene
#     plot_data(logOfK, accuracyValuesOfGolubArcene, accuracyValuesOfRFEArcene, accuracyValuesOfModifiedArcene, "Arcene Dataset")

        
#     print "Best Combination Golub", best_golub_answer
#     print "Best Global Results", best_global_answer

    
#         print grid_search_golub.best_params_
#     gotAccuracy = np.mean(cross_validation.cross_val_score(pipeline_golub, X, y, cv=cross_valid, scoring='accuracy'))
#         print "Accuracy Using Golub", gotAccuracy
#         return gotAccuracy
#     k_rfe = 512
#     selector_rfe = 
    
#     k_l1Modified = 512
    
#     selector_l1_modified = SelectKBest(L1_Modified_Features, k_l1Modified)
    
#     X_L, y_L = generate_data(2)
#     kValuesLGolub, accuracyValuesOfGolubL =golub_feature_select(X_L, y_L, 2)


#     kValuesLRFE, accuracyValuesOfRFEL = rfe_feature_select(X_L, y_L, 2)
#     kValuesLModified, accuracyValuesOfModifiedL = modified_L1_feature_select(X_L, y_L, 2)
#     logOfKL = np.log2(kValuesLGolub)

#     plot_data(logOfKL, accuracyValuesOfGolubL, accuracyValuesOfRFEL, accuracyValuesOfModifiedL, "Leukemia Dataset")

#     compute_accuracy_of_selector(golub_score, "Golub", "l2", 100, X_L, y_L)
#     golub_rank_L = golub_score(X_L, y_L, 1)


# In[ ]:



