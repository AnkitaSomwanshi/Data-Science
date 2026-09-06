import numpy as np             # Try using pandas
from sklearn.neighbors import KNeighborsClassifier

def main():

    # Independent
    X = np.array([[1,2],[2,3],[3,1],[5,6]])

    # Dependent
    Y = np.array(["Red","Red","Blue","Blue"])

    new_point = np.array([[3,3]])

    print("Independent Variable are : ")
    print(X)

    print("dependent Variable are : ")
    print(Y)

    print("Testing point is : ")
    print(new_point)

if __name__ == "__main__":
    main()