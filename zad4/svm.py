from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from wineDataFromCSV import getWinesDatasetFromCSV
from wineDataFromCSV import binaryClassification

def loadDataset():
    data, quality = getWinesDatasetFromCSV()
    X_train, X_test, y_train, y_test = train_test_split(data, quality, test_size=0.3)

    y_train = binaryClassification(y_train)
    y_test = binaryClassification(y_test)

    return [X_train, X_test, y_train, y_test]

def svm(kernel, c, gamma, dataSet):
    model = SVC(kernel=kernel, C=c, gamma=gamma)
    model.fit(dataSet[0], dataSet[2])
    y_pred = model.predict(dataSet[1])

    return metrics.accuracy_score(dataSet[3], y_pred)

# Wypisuje na wyjściu standardowym listę wartości accuracy dla konkretnych parametrów C i gamma
def svmLogs():
    kernels = ['linear', 'rbf']
    constants = [0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1]
    data = loadDataset()
    for ker in kernels:
        print('\nKernel type: ', ker, '\n')
        for c in constants:
            for g in gammas:
                print(f'C= {c}'.ljust(7), f'gamma= {g}'.ljust(13), 'accuracy=', round(svm(ker, c, g, data),3))

# Wypisuje na wyjściu standardowym tabelkę accuracy w zależności od parametrów C i gamma
def svmTable():
    kernels = ['rbf']
    constants = [0.01, 0.1, 1, 10, 100]
    gammas = [1e-4, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    data = loadDataset()
    results = []

    for ker in kernels:
        print('\nKernel type: ', ker, '\n')
        for i in range(len(constants)):
            const_res = []
            for j in range(len(gammas)):
                c = constants[i]
                g = gammas[j]
                const_res.append(round(svm(ker, c, g, data),3))
            results.append(const_res)
        top = 'C\gamma'.ljust(10)
        for g in gammas:
            top +=  str(g).ljust(10)
        print(top)
        for i in range(len(constants)):
            line = str(constants[i]).ljust(10)
            for j in results[i]:
                line += str(j).ljust(10)
            print(line)
    print()

def main():
    # svmLogs()
    svmTable()

    # Pojedyńcze wykonanie funkcji dla zadanych parametrów
    # data = loadDataset()
    # print(round(svm(kernel='rbf', c=1e6, gamma=1e-6, dataSet=data),3))

if __name__ == '__main__':
    main()
