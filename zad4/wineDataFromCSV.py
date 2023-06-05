import csv

path_r = "zad4/winequality-red.csv"
path_w = "zad4/winequality-white.csv"

def getWinesDatasetFromCSV(kind='r'):
    if kind == 'r':     path = path_r
    elif kind == 'w':   path = path_w
    else:               raise Exception("Incorrect wine type. Chose 'r' for red wine, 'w' for white wine")

    with open(path, 'r') as file_handle:
        reader = csv.reader(file_handle, delimiter=';')
        next(reader, None)      # Pomija pierwszy wiersz z nagłówkami
        data_train = []
        quality_train = []
        for row in reader:
            atributes = []
            for a in row:
                atributes.append(float(a))
            data_train.append(atributes[:-1])
            quality_train.append(atributes[-1])
        return data_train, quality_train

def binaryClassification(y_table):
    bin_y_train = []
    for y in y_table:
        if y > 5:   bin_y_train.append(1)
        else:       bin_y_train.append(0)
    return bin_y_train
