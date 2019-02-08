# \exercises\src\classesandobjects02.py
class Kreis:
    def __init__(self, radius, farbe):
        self.radius = radius
        self.farbe = farbe

    def getKreis(self):
        return self.radius, self.farbe

class Erweiterung:
    def __init__(self, radius, farbe, xposition):
        Kreis.__init__(self, radius, farbe)
        self.xposition = xposition

    def getKreis(self):
        print(Kreis.getKreis(self), self.xposition)
        return Kreis.getKreis(self), self.xposition

kreis1 = Erweiterung(3, "weiß", 4)
kreis1.getKreis()