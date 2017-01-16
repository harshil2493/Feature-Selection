
# coding: utf-8

# In[4]:

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


def L1_SVM(X, y, flag):
    print "**Calculating Number Of None Zero Weight Features**"
    if flag == 1:
        print "For Arcene Dataset"
    else:
        print "For Leukemia Dataset"
    run_svm = 1
    none_zero_counters = []
    score_vector_fold = np.zeros(X.shape[1])

    
    for count_down in range((run_svm)):
        print "Iteration: ", count_down
        classifier = LinearSVC(penalty='l1', dual=False)
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
        for i in range(len(classifier.coef_[0])):
    #         print coef
            if(classifier.coef_[0][i] !=0):
                counter_of_needed_feature = counter_of_needed_feature + 1
                score_vector_fold[i] = 1

    #     print classifier.coef_
#         print "Total Non Zero Weight Features", counter_of_needed_feature
        none_zero_counters.append(counter_of_needed_feature)
#     print "Average Number Of Non Zero ", round(np.average(none_zero_counters))
    return score_vector_fold
#     print "Score Vector", score_vector_fold
#     for i in score_vector_fold:
#         print i

    #     print "Range", round(np.min(none_zero_counters)), " To ", round(np.max(none_zero_counters))
    

def generateKSubSample(X, y, flag):
    k = 0
    if flag == 1:
        k = 8
    else:
        k = 2
    print "K Will Be", k
    number_rows = X.shape[0]
    number_Train = int(round(number_rows*0.8))
    number_Test = number_rows - number_Train
#     print number_Train,number_Test,number_Train+number_Test
    
    rows = np.arange(number_rows)
    np.random.shuffle(rows)
#     print rows
#     print rows
    
    training = rows[:number_Train]
    testing = rows[number_Train:]
#     print "y", y
    y = y.reshape((-1,1))
#     print y
    X_training = X[training,:]
    y_training = y[training,:]
    X_testing = X[testing,:]
    y_testing = y[testing,:]
    
    y_training = y_training.reshape(1, -1)[0]
#     print y
#     print y_training
#     print "Random",len(X_training)
    
    X_split_data = np.split(X_training, k)
    y_split_data = np.split(y_training, k)
#     print "Before Split", X_training
#     print X_split_data[0].shape
    return X_split_data, y_split_data
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
    X_Arcene,y_Arcene = generate_data(1)
    
    randomization = 10
    final_counter_arcene = []
#     print X_Arcene
#     print y_Arcene
    for i in range(randomization):
        print "For Randomization", i
        X_Arcene_Split, y_Arcene_Split = generateKSubSample(X_Arcene, y_Arcene, 1)

    #     print "X", X_Arcene_Split[0]
    #     print "y", y_Arcene_Split[0]
        score_vector_arcene = np.zeros(X_Arcene.shape[1])
        for i in range(len(X_Arcene_Split)):
    #         print "X", X_Arcene_Split[i]
    #         print "y", y_Arcene_Split[i]
            print "Fold: ", i
            score_vector_arcene = score_vector_arcene + L1_SVM(X_Arcene_Split[i], y_Arcene_Split[i], 1)
        counter = 0
        for rank in score_vector_arcene:
            if rank !=0:
                counter += 1
    #             print rank
    #         else:

        print "Non Zero Weight For Arcene", counter
        final_counter_arcene.append(counter)
    print "Average Non Zero Weights For Arcene Dataset", round(np.average(final_counter_arcene))
    print "Range", np.min(final_counter_arcene), " To ", np.max(final_counter_arcene)
    X_L, y_L = generate_data(2)
    final_counter_L = []
    
    print "For Leukemia Dataset"

    for i in range(randomization):
        print "For Randomization", i

        X_L_Split, y_L_Split = generateKSubSample(X_L, y_L, 2)
        score_vector_L = np.zeros(X_L.shape[1])

        for i in range(len(X_L_Split)):
            print "Fold: ", i
            score_vector_L = score_vector_L + L1_SVM(X_L_Split[i], y_L_Split[i], 2)

        counter = 0
        for rank in score_vector_L:
            if rank !=0:
                counter += 1
        
    #             print rank
    #         else:
        final_counter_L.append(counter)
#         print "Non Zero Weight For Leukemia Dataset", counter
    print "Average Non Zero Weights For Leukemia Dataset", round(np.average(final_counter_L))
    print "Range", np.min(final_counter_L), " To ", np.max(final_counter_L)



# In[ ]:



