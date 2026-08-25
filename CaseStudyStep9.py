import pandas as pd

import matplotlib.pyplot as plt  #  -->Visualisation
import seaborn as sns            #    

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
                                          
Border = "-"*30                           

##############################################################
# step 1 : Load the data set
##############################################################

print(Border)
print("step 1 : Load the dataset")
print(Border)

DataPath = "iris.csv"

df = pd.read_csv(DataPath)

print("Dataset loaded succesfully")

print("Initial Entries from Dataset are : ")
print(df.head())

##############################################################
# step 2 : Exploratory Data Analysis (EDA)
##############################################################

print(Border)
print("step 2 : Exploratory Data Analysis (EDA)")
print(Border)

print("Shape of Dataset : ",df.shape)

print("Column names : ",list(df.columns))

print("Missing values per column : ")
print(df.isnull().sum())                  # Canonical Function call

print("Class distribution (species count)")
print(df["species"].value_counts())

print("Statistical report of dataset : ")
print(df.describe())

##############################################################
# step 3 : Decide Independent & Dependent Variables
##############################################################

print(Border)
print("step 3 : Decide Independent & Dependent Variables")
print(Border)

# X : Independent variables / Features
# Y : Dependent Variables / Labels

feature_cols = ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]

X = df[feature_cols]
Y = df["species"]

print("X Shape : ",X.shape)
print("Y Shape : ",Y.shape)

##############################################################
# step 4 : Visualisation of Dataset
##############################################################

print(Border)
print("step 4 : Visualisation of Dataset")
print(Border)

# Scatter plot
plt.figure(figsize = (7,5))

for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(temp["petal length (cm)"],temp["petal width (cm)"],label = sp)

plt.title("Marvellous Iris Case Study")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.legend()
plt.grid()
plt.show()

##############################################################
# step 5 : Split the dataset for training and testing
##############################################################

print(Border)
print("step 5 : Split the dataset for training and testing")
print(Border)

X_train, X_test, Y_train, Y_test  = train_test_split(X,Y,test_size=0.5,random_state=42)

print("DataSet Splitting Activity Done")

print("X : ",X.shape)   # (150,4)
print("Y : ",Y.shape)   #(150,)

print("X_train : ",X_train.shape)    # (75,4)
print("X_test : ",X_test.shape)      # (75,4)

print("Y_train : ",Y_train.shape)    # (75,)
print("Y_test : ",Y_test.shape)      # (75,)

##############################################################
# step 6 : Build the Model
##############################################################

print(Border)
print("step 6 : Build the Model")
print(Border)

model = DecisionTreeClassifier(max_depth=5)

print("Model gets Created Sucessfully")

##############################################################
# step 7 : Train the Model
##############################################################

print(Border)
print("step 7 : Train the Model")
print(Border)

model.fit(X_train,Y_train)

print("Model trained Sucessfully")

##############################################################
# step 8 : Test the Model
##############################################################

print(Border)
print("step 8 : Test the Model")
print(Border)

Y_pred = model.predict(X_test)

print("Model testing Done")

print("Expected Answers : ")
print(Y_test)

print("Predicted Answers : ")
print(Y_pred)

##############################################################
# step 9 : Evaluate the Model Performance
##############################################################

print(Border)
print("step 9 : Evaluate the Model")
print(Border)

accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy of model is : ", accuracy*100)

print("Confusion Matrix")
print(confusion_matrix(Y_test,Y_pred))

print("Classification Report")
print(classification_report(Y_test,Y_pred))