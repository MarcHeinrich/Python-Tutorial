# src/FileHandlingReadWrite.py
# Erstellen, Oeffnen und Schliessen der datei.txt

file = open("datei.txt", "x")
file.close()

# src/FileHandlingReadWrite.py
# datei.txt Text hinzufuegen

file = open("datei.txt", "w")
file.write("Hallo Welt.")  # eine Zeile hinzufuegen
file.close()

# src/FileHandlingReadWrite.py
# datei.txt Text hinzufuegen

textList = ["Hallo Welt.\n", "Das ist ein...\n", "Beispieltext"]
# Ohne \n wuerde der gesamte Inhalt in eine Zeile geschrieben

file = open("datei.txt", "w")
file.writelines(textList)
file.close()

# src/FileHandlingReadWrite.py
# Auslesen und Ausgeben des Inhalts der datei.txt

file = open("datei.txt", "r")
print(file.read())
file.close()

# Ausgabe:
# Hallo Welt.
# Das ist ein
# Beispieltext

# src/FileHandlingReadWrite.py
# Ausgeben des Inhalts der datei.txt

file = open("datei.txt", "r")
for line in file:
    print(line)
file.close()

# Ausgabe
# Hallo Welt.
#
# Das ist ein
#
# Beispieltext

# src/FileHandlingReadWrite.py
# Auslesen und Ausgeben des Inhalts der datei.txt

file = open("datei.txt", "r")
print(file.readlines())
file.close()

# Ausgabe: ['Hallo Welt.\n', 'Das ist ein...\n', 'Beispieltext']

# src/FileHandlingReadWrite.py
# Ausgeben der Zeigerposition
file = open("datei.txt", "r")
print(file.tell())
print(file.readlines())
print(file.tell())
print(file.readlines())
file.close()

# Ausgabe:
# 0
# ['Hallo Welt.\n', 'Das ist ein...\n', 'Beispieltext']
# 41
# []

# src/FileHandlingReadWrite.py
# Zeiger zuruecksetzten

file = open("datei.txt", "r")
print(file.tell())
print(file.readlines())
file.seek(0)
print(file.tell())
print(file.readlines())
file.close()

# Ausgabe:
# 0
# ['Hallo Welt.\n', 'Das ist ein...\n', 'Beispieltext']
# 0
# ['Hallo Welt.\n', 'Das ist ein...\n', 'Beispieltext']

# src/FileHandlingReadWrite.py
# with-Statement zum Oeffnen der Datei

with open("datei.txt", "r") as file:
    print(file.read())

# Ausgabe:
# Hallo Welt.
# Das ist ein...
# Beispieltext