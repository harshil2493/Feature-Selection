
# coding: utf-8

# In[2]:

import numpy as np
from matplotlib import pyplot as plt
import bz2
import numpy as np
from sklearn.datasets import svmlight_format
from scipy.sparse import csr_matrix

def golub_score(X, y):
    flag = 1
    if(flag ==1):
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

        print "Positive Datasize",X_Positive.shape
        print "Negative Datasize",X_Negative.shape
        print "Positive Labels",y_Positive.shape
        print "Positive Labels",y_Negative.shape

        Average_Positive_Example = np.average(X_Positive, axis = 0)
        Average_Negative_Example = np.average(X_Negative, axis = 0)

        Standard_Positive_Example = np.std(X_Positive, axis=0)
        Standard_Negative_Example = np.std(X_Negative, axis=0)

        print "Average Positive", Average_Positive_Example.shape
        print "Average Negative", Average_Negative_Example.shape
        print "Standard Positive", Standard_Positive_Example.shape
        print "Standard Negative", Standard_Negative_Example.shape

        Average_Difference = np.absolute(Average_Positive_Example - Average_Negative_Example)
        Standard_Sum_Temporary = Standard_Positive_Example + Standard_Negative_Example

        Standard_Sum = []
        for x in Standard_Sum_Temporary:
#             print "X Value", x
            if(x == 0):
                x = 1
            Standard_Sum.append(x)

        Golub = Average_Difference / Standard_Sum
        
#     elif(flag==2):
#         Golub = y
# # #         X = np.asarray(X)
# #         print "Y Shape", y
#         X_Positive = []
#         X_Negative = []
#         y_Positive = []
#         y_Negative = []

#         for i in range(len(y)):
#             if(y[i] == -1):
#                 y_Negative.append(y[i])
#                 X_Negative.append(np.asarray(X[i][0]))
                
#             elif(y[i] == 1):
#                 y_Positive.append(y[i])
#                 X_Positive.append(np.asarray(X[i][0]))
                

#         X_Positive = np.asarray(X_Positive)
#         X_Negative = np.asarray(X_Negative)
#         y_Positive = np.asarray(y_Positive)
#         y_Negative = np.asarray(y_Negative)

#         print "Positive Datasize",X_Positive.shape
#         print "Negative Datasize",X_Negative.shape
#         print "Positive Labels",y_Positive.shape
#         print "Positive Labels",y_Negative.shape
        
#         Average_Positive_Example = np.average(X_Positive, axis = 0)
#         Average_Negative_Example = np.average(X_Negative, axis = 0)

#         Standard_Positive_Example = np.std(X_Positive, axis=0)
#         Standard_Negative_Example = np.std(X_Negative, axis=0)

#         print "Average Positive", Average_Positive_Example.shape
#         print "Average Negative", Average_Negative_Example.shape
#         print "Standard Positive", Standard_Positive_Example.shape
#         print "Standard Negative", Standard_Negative_Example.shape

#         Average_Difference = np.absolute(Average_Positive_Example - Average_Negative_Example)
#         Standard_Sum_Temporary = Standard_Positive_Example + Standard_Negative_Example

#         Standard_Sum = []
#         for x in Standard_Sum_Temporary:
# #             print "X Value", x
#     #         if(x == 0):
#     #             x = 1
#             Standard_Sum.append(x)

#         Golub = Average_Difference / Standard_Sum

    return Golub, Golub

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
#         print "X_Train", X_train.shape
        
#         print X_train
        X_test = data_test[0].toarray()
        y_test = data_test[1]
#         print X_train.shape
#         print y_train.shape

        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        print "After Concate: X Size: ", X.shape
        print "After Concate: y Size: ", y.shape
#         print X
    return X, y
        
if __name__=='__main__' :
    X_Arcene,y_Arcene = generate_data(1)
#     print X_Arcene[0].shape
# 
    golub_rank_arcene = golub_score(X_Arcene, y_Arcene)
    print golub_rank_arcene
    
    X_L, y_L = generate_data(2)
#     X_Final = []
#     for x in X_L:
# #         x = x[0,:]
# #         print x
#         x = np.array(x)
#         x = np.squeeze(x)
# #         print x
# #         print x.shape
#         X_Final.append(x)
        
#     X_Final = np.asarray(X_Final)
#     print X_Final.shape
    
    
    golub_rank_L = golub_score(X_L, y_L)
    
    print golub_rank_L
#     print X_L
    
#     print golub_rank

#     plot_graph(golub_rank)


# In[13]:

import numpy as np

X = [[1, 2],[35, 4],[5, 6],[7, 8]]
print np.average(X, axis = 1)


# In[ ]:



