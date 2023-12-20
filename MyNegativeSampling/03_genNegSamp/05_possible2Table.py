def readCsv(filename):
    data = []
    with open(filename,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            line = line.split(",")
            data.append(line)
    return data

BacteiaPossiblePath = 'data/possible/Bac_update.csv'
BacteriaPathogenHash = readCsv("data/Hash/BacteriaPathogenHash.csv")
BacteriaHostHash = readCsv("data/Hash/BacteriaHostHash.csv")

BacPossibleTable = readCsv(BacteiaPossiblePath)
with open("data/possible/BacPossibleTable.csv", "w") as f:
    for line in BacPossibleTable:
        for i in BacteriaPathogenHash:
            if line[0] == i[1]:
                f.write(i[0])

        f.write(',')

        for j in BacteriaHostHash:
            if line[1] == j[1]:
                f.write(j[0])
        f.write('\n')

'''
VirusPossiblePath = 'data/possible/Vir_update.csv'
VirusPathogenHash = readCsv("data/Hash/VirusPathogenHash.csv")
VirusHostHash = readCsv("data/Hash/VirusHostHash.csv")

VirPossibleTable = readCsv(VirusPossiblePath)
with open("data/possible/VirPossibleTable.csv", "w") as f:
    for line in VirPossibleTable:
        for i in VirusPathogenHash:
            if line[0] == i[1]:
                f.write(i[0])

        f.write(',')

        for j in VirusHostHash:
            if line[1] == j[1]:
                f.write(j[0])
        f.write('\n')
        '''