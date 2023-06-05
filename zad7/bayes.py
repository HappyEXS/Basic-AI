from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import statistics

from read_dataset import *

''' Wartosci numeryczne klas:
-- Iris Setosa      = 0
-- Iris Versicolour = 1
-- Iris Virginica   = 2
'''

class NaiveBayes:
    def __init__(self, number_of_classes, number_of_attributes):
        '''Przypisuję liczby klas i liczby parametrów, aby w metodach móc iterować przez klasy i parametry.
        Zarówno klasy i atrybuty są ponumerowane 1, 2, 3, itd. w ten sposób można się do nich odwoływać.'''
        self.attributes = number_of_attributes  # liczba atrybutów
        self.classes = number_of_classes        # liczba klas
        self.class_prob = []    # Prawdopodobieństro i-tej klasy
        self.train_set = []     # Zbiór trenujący dla i-tej klasy, dla j-tego atrybutu - self.train_set[i][j]

        for i in range(self.classes):
            self.train_set.append([])
            for j in range(self.attributes):
                self.train_set[i].append([])


    def calculate_prob_of_class(self, class_number, train_output):
        '''Zwraca prawdopodobieństwo klasy w zbiorze trenującym'''
        return train_output.count(class_number) / len(train_output)

    def fit(self, train_input, train_output):
        '''Funkcja do załadowania danych trenujących'''
        # Załadowanie wartoci self.class_prob
        for i in range(self.classes):
            self.class_prob.append(self.calculate_prob_of_class(i, train_output))

        # Zaadowanie wartosci  self.train_set
        for i in range(len(train_output)):
            for j in range(self.attributes):
                self.train_set[train_output[i]][j].append(train_input[i][j])


    def calculate_single_cond_prob(self, class_number, parameter_number, x):
        '''Zwraca prawdopodobieństwo warunkowe x|klasy(class_number)parametru(parameter_number)'''
        # Kożystam z normalnego rozkładu prawdopodobieństwa
        average = np.average(self.train_set[class_number][parameter_number])
        standard_deviation = statistics.stdev(self.train_set[class_number][parameter_number])
        probability = 1/(standard_deviation * np.sqrt(2* np.pi))
        probability *= np.exp(-np.power(x-average, 2)/(2* np.power(standard_deviation, 2)))
        return probability

    def calculate_cond_prob(self, class_number, X):
        '''Zwraca prawdopodobieństwo warunkowe X|klasy(class_number)'''
        probability = 1
        for i in range(len(X)):
            probability *= self.calculate_single_cond_prob(class_number, i, X[i])
        return probability

    def predict(self, X):
        # Obliczam prawdopodobieństwo każdej kalsy
        prob_of_class = []
        for i in range(self.classes):
            cond_prob = self.calculate_cond_prob(i, X)
            prob_of_class.append(self.class_prob[i] * cond_prob)

        # Wybieram klasę z największym prawdopodobieństwem
        return np.argmax(prob_of_class)

def test():
    '''Fukncja do testowania algorytmu w zależnosci od test_size'''
    for test_size in [0.05, 0.1, 0.2, 0.3, 0.4]:
        acc = []
        for i in range(100):
            X_train, X_test, y_train, y_test = get_data(test_size)

            bayes = NaiveBayes(3, 4)
            bayes.fit(X_train, y_train)

            y_pred = [bayes.predict(x) for x in X_test]

            accuracy = metrics.accuracy_score(y_test, y_pred)
            acc.append(accuracy)
        print(f"\nTest_size={test_size} Average accuracy :", round(np.average(acc), 3))
    print()


def main():
    # test()
    X_train, X_test, y_train, y_test = get_data(0.3)

    bayes = NaiveBayes(3, 4)
    bayes.fit(X_train, y_train)

    y_pred = [bayes.predict(x) for x in X_test]

    accuracy = metrics.accuracy_score(y_test, y_pred)

    print(accuracy)
    plot_confusion_matrix(y_test, y_pred, "naive Bayes", accuracy)


if __name__ == "__main__":
    main()