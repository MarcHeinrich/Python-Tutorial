# \exercises\src\classesandobjects01
class Kreis:
    def __init__(self, radius, farbe):
        self.radius = radius
        self.farbe = farbe

    def getKreis(self):
        print(self.radius, self.farbe)

kreis1 = Kreis(5, "gelb")
kreis2 = Kreis(3, "blau")
kreis1.getKreis()
kreis2.getKreis()