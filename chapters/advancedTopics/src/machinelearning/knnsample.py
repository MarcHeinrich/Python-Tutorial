import numpy as numpy
from sklearn import datasets


# Label festlegen
classLabelVector = ["label1", "label2", "label3"]
errorCount = 0
# k-Eingrenzung (hier: auf 5 Nachbarn einschränken)
k = 3


def classify(inX, dataSet, labels, k):
    rowCount = dataSet.shape[0]  # Anzahl an Zeilen bestimmen

    diffMat = numpy.tile(inX, (rowCount, 1)) - dataSet  # Berechnung der Katheten
    # (über tile() wird der Eingangsdatensatz über die Zeilenanzahl des dataSet vervielfacht,
    # der dataSet davon substrahiert)

    sqDiffMat = diffMat ** 2  # Quadrat der Katheten
    sqDistances = sqDiffMat.sum(axis=1)  # Aufsummieren der Differenzpaare
    distances = sqDistances ** 0.5  # Quadratwurzel über alle Werte
    sortedDistIndicies = distances.argsort()  # Aufsteigende Sortierung

    classCount = {}

    # print("inX = %s, k = %s" % (inX, k))
    # print(sortedDistIndicies)

    for i in range(k):  # Eingrenzung auf k-Werte in der sortierten Liste
        closest = labels[sortedDistIndicies[i]]  # Label (Kategorie entsprechend der Sortierung aufnehmen
        classCount[closest] = classCount.get(closest, 0) + 1  # Aufbau eines Dictionary über die

    sortedClassCount = sorted(classCount, key=classCount.get,
                              reverse=True)  # Absteigende Sortierung der gesammelten Labels in k-Reichweite
    # wobei die Sortierung über den Count (Value) erfolgt

    # print(classCount)
    # print(sortedClassCount[0])

    return sortedClassCount[0]  # Liefere das erste Label zurück
    # also das Label mit der höchsten Anzahl innerhalb der k-Reichweite


# Daten laden
dataSet = dt = datasets.load_iris()

# Anzahl Zeilen
rowCount = dataSet.size

numTestVectors = 30  # Datensätze 0 - 29 werden zum testen von k verwendet,
# die Datensätze ab Zeile 30 werden zur Klassifikation verwendet


for i in range(0, numTestVectors):  # Aufruf des Klassifikators von 0 bis 29

    result = classify.classify(dataSet[i, :], dataSet[numTestVectors:rowCount, :],
                               classLabelVector[numTestVectors:rowCount], k)

    print("%s - the classifier came back with: %s, the real answer is: %s" % (i, result, classLabelVector[i]))

    if result != classLabelVector[i]:
        errorCount += 1.0

print("Error Count: %d" % errorCount)
