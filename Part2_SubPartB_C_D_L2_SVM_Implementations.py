
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
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import SVC,LinearSVC

def L2_SVM_Selects_L2_Accuracy(X, y, flag):
   
    print "**Calculating Accuracy Of L2_SVM By Selecting Feature By L2**"
    print ""
    print "L2 SVM On Feature Selected By L2"
    if flag == 1:
        print "For Arcene Dataset"
    else:
        print "For Leukemia Dataset"
        
    run_svm = 1
    accuracy = []
    for count_down in range((run_svm)):
        print "Iterations: ", count_down
        classifier = LinearSVC(penalty='l2', dual=False)



        cross_valid = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
        selector = RFECV(classifier, step=0.1)
        selector = selector.fit(X, y)
        pipeline = make_pipeline(selector, classifier)
        got_accuracy = np.mean(cross_validation.cross_val_score(pipeline, X, y, cv=cross_valid, scoring='accuracy'))
#         got_accuracy = np.mean(cross_validation.cross_val_score(classifier, X, y, cv=cross_valid))

        accuracy.append(got_accuracy)
        print "Accuracy", got_accuracy
    print "Average Accuracy", np.average(accuracy)

def L2_SVM_Accuracy_All(X, y, flag):
   
    print "**Calculating Accuracy Of L2_SVM**"
    print ""
    print "L2 SVM On All Features"
    if flag == 1:
        print "For Arcene Dataset"
    else:
        print "For Leukemia Dataset"
        
    run_svm = 1
    accuracy = []
    for count_down in range((run_svm)):
        print "Iterations: ", count_down
        classifier = LinearSVC(penalty='l2', dual=False)
        
        classifier = classifier.fit(X, y)


#         selector = RFE(classifier, step=0.1)
#         selector = selector.fit(X, y)

        cross_valid = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)

#         rfe_svm = make_pipeline(selector, classifier)
#         got_accuracy = np.mean(cross_validation.cross_val_score(rfe_svm, X, y, cv=cross_valid))
        got_accuracy = np.mean(cross_validation.cross_val_score(classifier, X, y, cv=cross_valid, scoring='accuracy'))

        accuracy.append(got_accuracy)
        print "Accuracy", got_accuracy
    print "Average Accuracy", np.average(accuracy)

        
def L1_SVM_Selects_L2_Accuracy(X, y, flag):
    print "**Calculating Feature Selection Of L1_SVM**"
    print ""
    print "L2 SVM On Feature Selected By L1"
    if flag == 1:
        print "For Arcene Dataset"
    else:
        print "For Leukemia Dataset"
        
#     run_svm = 25
#     accuracy = []
#     for count_down in range((run_svm)):

    classifier = LinearSVC(penalty='l1', dual=False)

    cross_valid = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)

        
    selector = RFECV(classifier, step=0.1)
    selector = selector.fit(X, y)
    
# #     print selector.support_
# #     print selector.ranking_
#     print "Feature Selected",selector.n_features_
    
#     X_transformed=selector.transform(X)
    
#     print "Old X Shape: ", X.shape
#     print "New X Shape: ", X_transformed.shape
    
    classifier_L2 = LinearSVC(penalty = 'l2', dual=False)

    pipeline = make_pipeline(selector, classifier_L2)
#     print pipeline
    got_accuracy = np.mean(cross_validation.cross_val_score(pipeline, X, y, cv=cross_valid, scoring='accuracy'))
# #     accuracy.append(got_accuracy)
    print "Accuracy", got_accuracy
#     print "Average Accuracy", np.average(accuracy)
def L2_SVM(X, y, flag):
    print "**Calculating Number Of None Zero Weight Features**"
    if flag == 1:
        print "For Arcene Dataset"
    else:
        print "For Leukemia Dataset"
    run_svm = 25
    none_zero_counters = []
    for count_down in range((run_svm)):
        print "Iteration: ", count_down
        classifier = LinearSVC(penalty='l2', dual=False)
    #     selector = RFE(classifier, step=0.1)

    #     selector = selector.fit(X, y)
        classifier = classifier.fit(X, y)
    #     print "Answer", answer
    #     print selector.support_.shape
    #     print selector.ranking_.shape
        counter_of_needed_feature = 0
    #     for support in selector.support_:
    #         print support
    #         if(support == True):
    #             counter_of_needed_feature = counter_of_needed_feature+1

    #     print "Total Needed Features", counter_of_needed_feature

    #     for rank in selector.ranking_:
    #         print rank
        for coef in classifier.coef_[0]:
    #         print coef
            if(coef !=0):
                counter_of_needed_feature = counter_of_needed_feature + 1

    #     print classifier.coef_
        print "Total Non Zero Weight Features", counter_of_needed_feature
        none_zero_counters.append(counter_of_needed_feature)
    print "Average Number Of Non Zero ", round(np.average(none_zero_counters))
    print "Range", round(np.min(none_zero_counters)), " To ", round(np.max(none_zero_counters))
    

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
        
#         print X

        
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
        
#         print X

    return X, y
        
if __name__=='__main__' :
#     print "Starting Execution"
#     print "Working On Arcene"
#     X_Arcene,y_Arcene = generate_data(1)
# #     print "X_Arcene Size", X_Arcene.shape
    
# #     L2_SVM(X_Arcene, y_Arcene, 1)

#     L1_SVM_Selects_L2_Accuracy(X_Arcene, y_Arcene, 1)
#     L2_SVM_Accuracy_All(X_Arcene, y_Arcene, 1)
#     L2_SVM_Selects_L2_Accuracy(X_Arcene, y_Arcene, 1)
#     print "Done On Arcene"
#     print "Working On Leukemia"
    X_L, y_L = generate_data(2)
#     L2_SVM(X_L, y_L, 2)

    L1_SVM_Selects_L2_Accuracy(X_L, y_L, 2)
    
    L2_SVM_Accuracy_All(X_L, y_L, 2)
    L2_SVM_Selects_L2_Accuracy(X_L, y_L, 2)
    print " Whole Program Done!..."



# 
