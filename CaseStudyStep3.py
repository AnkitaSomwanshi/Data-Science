import pandas as pd                       
                                          
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


