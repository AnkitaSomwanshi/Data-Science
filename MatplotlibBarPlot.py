import matplotlib.pyplot as plt

def main():

    language = ["C","C++","Java","Python"]
    students = [30,40,35,55]

    plt.bar(
        language,
        students,
        width =0.6,             # Width of Bars
        edgecolor = "Black",    # Border color of bars
        linewidth = 1,          # width of bar border
        alpha = 0.8,            # Transperance 0.0 to 1.0
        label = "Students"      # legend text
    ) 

    plt.title("Marvellous Bar Plot")
    plt.xlabel("Languages")
    plt.ylabel("Number of students")
    
    #plt.grid(True)
    
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()