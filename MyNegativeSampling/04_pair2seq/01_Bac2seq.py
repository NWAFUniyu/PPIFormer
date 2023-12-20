import os
import csv
import random
import pandas as pd
from Bio import SeqIO
def myReadCsv(filename):
    data = []
    with open(filename,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            line = line.split(",")
            data.append(line)
    return data


BacteriaSeq = 'seqAll/BacteriaPathogenFinal(Human)X.fasta'
HomoSeq = 'seqAll/BacteriaHumanFinalX.fasta'
'''
Bacteria_Species = ["Bacillus anthracis","Burkholderia mallei","Francisella tularensis","Yersinia pestis","Others"]

for Bacteria in Bacteria_Species:
    train_dir = 'dataset/train/bacteria/species/%s/'%Bacteria
    res_train = 'Final/train/bacteria/species/%s/'%Bacteria
    os.makedirs(res_train , exist_ok=True)

    # 读取UniProt ID的CSV文件
    species1_ids = []
    species2_ids = []
    index_ids = []
    BacteriaRecord = []
    HomoRecord = []
    with open('dataset/train/bacteria/species/'+  Bacteria +'/'+Bacteria+'_train.csv') as f:
        for line in f:
            columns = line.strip().split(',')
            species1_ids.append(columns[0])
            species2_ids.append(columns[1])
            index_ids.append(columns[2])

    with open(BacteriaSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            BacteriaRecord.append(record)

    with open(res_train + Bacteria+'_Pathogen.fasta', 'w') as f:
        for i in species1_ids:
            for j in BacteriaRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')


    with open(HomoSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            HomoRecord.append(record)

    with open(res_train + Bacteria+'_Homo.fasta', 'w') as f:
        for i in species2_ids:
            for j in HomoRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')

    with open(res_train + Bacteria+'_label.txt', 'w') as f:
        for i in index_ids:
            f.write(i)
            f.write("\n")

for Bacteria in Bacteria_Species:
    test_dir = 'dataset/test/bacteria/species/%s/'%Bacteria
    res_test = 'Final/test/bacteria/species/%s/'%Bacteria
    os.makedirs(res_test , exist_ok=True)
    # 读取UniProt ID的CSV文件
    species1_ids = []
    species2_ids = []
    index_ids = []
    BacteriaRecord = []
    HomoRecord = []
    with open('dataset/test/bacteria/species/'+  Bacteria +'/'+Bacteria+'_test.csv') as f:
        for line in f:
            columns = line.strip().split(',')
            species1_ids.append(columns[0])
            species2_ids.append(columns[1])
            index_ids.append(columns[2])

    with open(BacteriaSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            BacteriaRecord.append(record)

    with open(res_test + Bacteria+'_Pathogen.fasta', 'w') as f:
        for i in species1_ids:
            for j in BacteriaRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')


    with open(HomoSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            HomoRecord.append(record)

    with open(res_test + Bacteria+'_Homo.fasta', 'w') as f:
        for i in species2_ids:
            for j in HomoRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')

    with open(res_test + Bacteria+'_label.txt', 'w') as f:
        for i in index_ids:
            f.write(i)
            f.write("\n")

'''
# #####All################################
species1_ids = []
species2_ids = []
index_ids = []
BacteriaRecord = []
HomoRecord = []
os.makedirs('Final/train/bacteria/all/' , exist_ok=True)
os.makedirs('Final/test/bacteria/all/' , exist_ok=True)
with open('dataset/train/bacteria/all/All_train.csv') as f:
    for line in f:
        columns = line.strip().split(',')
        species1_ids.append(columns[0])
        species2_ids.append(columns[1])
        index_ids.append(columns[2])

with open(BacteriaSeq) as f:
    for record in SeqIO.parse(f, 'fasta'):
        BacteriaRecord.append(record)

with open('Final/train/bacteria/all/All_Pathogen.fasta', 'w') as f:
    for i in species1_ids:
        for j in BacteriaRecord:
            if i in j.id:
                SeqIO.write(j, f, 'fasta')


with open(HomoSeq) as f:
    for record in SeqIO.parse(f, 'fasta'):
        HomoRecord.append(record)

with open('Final/train/bacteria/all/All_Homo.fasta', 'w') as f:
    for i in species2_ids:
        for j in HomoRecord:
            if i in j.id:
                SeqIO.write(j, f, 'fasta')

with open('Final/train/bacteria/all/All_label.txt', 'w') as f:
    for i in index_ids:
        f.write(i)
        f.write("\n")

#####All Test################################
species1_ids = []
species2_ids = []
index_ids = []
BacteriaRecord = []
HomoRecord = []
with open('dataset/test/bacteria/all/All_test.csv') as f:
    for line in f:
        columns = line.strip().split(',')
        species1_ids.append(columns[0])
        species2_ids.append(columns[1])
        index_ids.append(columns[2])

with open(BacteriaSeq) as f:
    for record in SeqIO.parse(f, 'fasta'):
        BacteriaRecord.append(record)

with open('Final/test/bacteria/all/All_Pathogen.fasta', 'w') as f:
    for i in species1_ids:
        for j in BacteriaRecord:
            if i in j.id:
                SeqIO.write(j, f, 'fasta')


with open(HomoSeq) as f:
    for record in SeqIO.parse(f, 'fasta'):
        HomoRecord.append(record)

with open('Final/test/bacteria/all/All_Homo.fasta', 'w') as f:
    for i in species2_ids:
        for j in HomoRecord:
            if i in j.id:
                SeqIO.write(j, f, 'fasta')

with open('Final/test/bacteria/all/All_label.txt', 'w') as f:
    for i in index_ids:
        f.write(i)
        f.write("\n")