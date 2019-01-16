import numpy as numpy
from sklearn import datasets


# Label festlegen
classLabelVector = ["label1", "label2", "label3"]
errorCount = 0
# k-Eingrenzung (hier: auf 3 Nachbarn einschränken)
k = 3
# Datensätze 0 - 29 werden zum testen von k, die 
# Datensätze ab Zeile 30 werden zur Klassifikation verwendet
numTestVectors = 30


def classify(inX, dataSet, labels, k):
    # Anzahl an Zeilen bestimmen
    rowCount = dataSet.shape[0]

    # Berechnung der Katheten
    diffMat = numpy.tile(inX, (rowCount, 1)) - dataSet
    # Quadrat der Katheten
    sqDiffMat = diffMat ** 2
    # Aufsummieren der Differenzpaare
    sqDistances = sqDiffMat.sum(axis=1)
    # Quadratwurzel über alle Werte
    distances = sqDistances ** 0.5
    # Aufsteigende Sortierung
    sortedDistIndicies = distances.argsort()

    classCount = {}

    # print("inX = %s, k = %s" % (inX, k))
    # print(sortedDistIndicies)

    # Eingrenzung auf k-Werte in der sortierten Liste
    for i in range(k):
        # Label (Kategorie mit Sortierung aufnehmen)
        closest = labels[sortedDistIndicies[i]]
        # Aufbau eines Dictionary
        classCount[closest] = classCount.get(closest, 0) + 1

    # Absteigende Sortierung der Labels in k-Reichweite
    # wobei die Sortierung über den Count (Value) erfolgt
    sortedClassCount = sorted(classCount, 
			key=classCount.get, reverse=True)

    # print(classCount)
    # print(sortedClassCount[0])

    # Liefere das erste Label zurück, also das Label
	# mit der höchsten Anzahl innerhalb der k-Reichweite
    return sortedClassCount[0]


# Daten laden
dataSet = dt = datasets.load_iris()

# Anzahl Zeilen
rowCount = dataSet.size

# Aufruf des Klassifikators von 0 bis 29
for i in range(0, numTestVectors):
    result = classify.classify(dataSet[i, :], 
		dataSet[numTestVectors:rowCount, :],
		classLabelVector[numTestVectors:rowCount], k)

    print("%s - the classifier came back with: %s, " +
		"the real answer is: %s" %
          (i, result, classLabelVector[i]))

    if result != classLabelVector[i]:
        errorCount += 1.0

print("Error Count: %d" % errorCount)
