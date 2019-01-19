# Bibliotheken einbinden
import numpy as numpy		
import matplotlib.pyplot as pyplot

# Daten aus Datei lesen
dataset, classLabelVector, classColorVector = 
	readDataSet("SampleFile.txt")
	
# Plot definieren
fig = pyplot.figure()
ax = fig.add_subplot(111)

# Plot mit Daten versorgen
ax.scatter(dataSet[:,0], dataSet[:,1], marker ='o',
	color=classColorVector)
	
# Achsenbeschriftungen festlegen
ax.set_xlabel("X-Label")
ax.set_ylabel("Y-Label")

# Achsenskala festlegen
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=0)

# Plot anzeigen
pyplot.show()