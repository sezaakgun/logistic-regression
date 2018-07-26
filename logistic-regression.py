# I dowloaded this data from -> https://www.kaggle.com/c/titanic
# The main idea is predict who would survive from titanic disaster
#
# Data looks like this
#       
#   PassengerId  Survived  Pclass                              Name              Sex   Age    SibSp  Parch            Ticket     Fare  Cabin  Embarked
#      1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171    7.2500    NaN      S
#      2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599   71.2833    C85      C
#      3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282    7.9250    NaN      S
#      4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803   53.1000   C123      S
#      5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450    8.0500    NaN      S
#   
#
#
#
#
#
############################################################################################################################################
##################################################                  LIBRARYS                  ##############################################
############################################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


############################################################################################################################################
##################################################              PREPROCESSING                 ##############################################
############################################################################################################################################

# Import dataset
data = pd.read_csv("titanic.csv")
print(data.head())
# Drop rows that contains NA (only age column has NA)
data = data.dropna(subset=["Age"])

# Drop cols that i think they are useless
data = data.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis="columns")

# Create dummy cols for categorical data
data = pd.get_dummies(data, columns=["Pclass","Sex"])
# print(data.head())

# Drop the useless dummy columns
# Ex: If its female , its score is 1. But if its 0 that means he is a man
# Therefore we dont need male column 
data = data.drop(["Pclass_3","Sex_male"],axis="columns")

# Actual no need for that but i like to see how high or low it can go
from sklearn.utils import shuffle
data = shuffle(data)


# Input data
inp_df = data.drop(["Survived"],axis="columns")
# Output data
out_df = data.drop(inp_df.columns.values,axis="columns")



# For feature scaling
# We use feature scaling for faster gradient decent approach
scaler = StandardScaler()
inp_df = scaler.fit_transform(inp_df)


# Split dataset for test and train
X_train, X_test, Y_train, Y_test = train_test_split(inp_df, out_df, test_size=0.2, random_state=42)




############################################################################################################################################
##################################################              FUNCTIONS                 ##################################################
############################################################################################################################################


def weight_initializer(n_features):

    # Initializing weights Z=w1.x1 + w2.x2 + ..... + wn.xn + b 
    W = np.zeros((1,n_features))
    b = 0

    return W,b



# This function maps the any input to (0,1) interval
# We use it because our cost function (cross-entrophy) takes probabilitys as inputs 
# Also it helps us determine what the final prediction is
# Search for sigmoid function if you want
def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result



# X is rowXfeature matrix
# Y is rowX1 vector
# W is 1Xfeature vector 
# b is scalar value
def update_weights(X,Y,W,b,learning_rate):

    # row count
    m = X.shape[0]

    # Z = x1.w1 + ...... + xn.wn + b
    Z = np.dot(W,X.T) + b

    # We predict the Y scores with current weights
    Y_prob = sigmoid_activation(Z)
    


    # Gradiant decent
    # This is the pre-calculated part
    # Look for cross-entrophy loss function 
    # This example based on this article 
    # https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc
    W = W - (1/m)*learning_rate*(np.dot(Y_prob-Y.T,X))
    b = b - (1/m)*learning_rate*(np.sum(Y_prob-Y.T,axis=1))

    # b came out as labeled data, because of that we use b[0] to access its value
    return W,b[0]




def start_learning(X,Y,W,b,learning_rate=0.001,iteration=5000):

    for i in range(iteration):
        W,b = update_weights(X,Y,W,b,learning_rate)

        # Every 1000 iteration give us a hint
        if i % 1000 == 0:
            print(i)

    return W,b



def predict(X,W,b):

    # Y = x1.w1 + ..... + xn.wn + la
    Y = np.dot(W,X.T) + b

    # Y_prob values are between 0 and 1
    Y_prob = sigmoid_activation(Y)

    # Create rowX1 zero matrix
    Y_pred = np.zeros((X.shape[0],1))
    for i,row_prob in enumerate(Y_prob[0]):  
        # If the predicted value is higher than 0.5 we predict its 1 else its stays 0
        if row_prob > 0.5:
            Y_pred[i] = 1
    return Y_pred








############################################################################################################################################
###########################################              ACTUAL LEARNING PART                ###############################################
############################################################################################################################################


# Number of features ( input columns )
n_features = X_train.shape[1]

W,b = weight_initializer(n_features)
W,b = start_learning(X_train,Y_train,W,b)
Y_pred2 = predict(X_test,W,b)
print("accuracy_score =",accuracy_score(Y_pred2,Y_test))





############################################################################################################################################
###########################################                  SKLEARN VERSION                 ###############################################
############################################################################################################################################




# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression ()
# lr.fit(X_train,Y_train.values.ravel())
# Y_pred1 = lr.predict(X_test)
# print("sklearn =",accuracy_score(Y_pred1,Y_test))



