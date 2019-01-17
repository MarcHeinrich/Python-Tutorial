# Importe
import numpy as numpy
from sklearn import datasets

dt_iris = datasets.load_iris()
iris = dt_iris.data[:, :4]

#TODO: Summary
print(dt_iris.summary())

#TODO: Head
print(dt_iris.head())

dataasny = numpy.array(iris)
