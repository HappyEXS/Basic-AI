import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_data(testSize):
    data = pd.read_csv('iris.data', sep=',', header=None)

    iris_type = data[4]
    iris_numeric = []
    for iris in iris_type:
        if iris.strip() == 'Iris-setosa': iris_numeric.append(0)
        elif iris == 'Iris-versicolor': iris_numeric.append(1)
        elif iris == 'Iris-virginica': iris_numeric.append(2)

    data = pd.DataFrame(data, columns=[0, 1, 2, 3])

    X_train, X_test, y_train, y_test = train_test_split(data.to_numpy(), iris_numeric, test_size=testSize)
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(test_y, predict_y, file_name, accuracy):
    labels = [0, 1, 2]
    C = confusion_matrix(test_y, predict_y, labels=labels)
    print("-"*20, "Confusion matrix", "-"*20)
    plt.figure(figsize=(5,4))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.text(2.2,3.35,f"Accuracy: {round(accuracy, 2)}")
    plt.title(label=file_name)
    plt.savefig(f'{file_name}.png')
    plt.show()