import pandas as pd                       # Pandas -  Series    DataFrame     Panel
                                          #             1D          2D         3D
Border = "-"*30                           #                                 Not Exist

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

