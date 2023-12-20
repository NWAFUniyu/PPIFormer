#----------------------------------------------------------
# Copyright (c) 2022, Northwest A&F University
# All rights reserved.
#
# File Name：01_genBacIntTable.py
#
# Abstract: This file is used to generate the interaction table based on the Hash table
# input：BacteriaInteract.csv
#        BacteriaHostHash.csv
#        BacteriaPathogenHash.csv
# output：BacteriaInteractionTable.csv
# Version：1.0
# Author：Yu Ni
# Finished Date：2022/12/5
# Instead version：None
#-----------------------------------------------------------
def readCsv(filename):
    data = []
    with open(filename,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            line = line.split(",")
            data.append(line)
    return data

IntractTable = readCsv("data/Bac/BacteriaInteract.csv")
HostHash = readCsv("data/Bac/BacteriaHostHash.csv")
PathogenHash = readCsv("data/Bac/BacteriaPathogenHash.csv")

# for line in HostHash:
#     print(line)
# for line in PathogenHash:
#     print(line)

with open('BacteriaInteractionTable.csv', "w") as f:
    for line in IntractTable:
        # print(line[0])
        for i in HostHash:
            if line[0] == i[0]:
                f.write(i[1])

        f.write(',')

        # print(line[1])
        for j in PathogenHash:
            if line[1] == j[0]:
                f.write(j[1])
        f.write('\n')