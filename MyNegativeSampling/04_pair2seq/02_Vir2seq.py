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


VirusSeq = 'seqAll/VirusPathogenFinalX.fasta'
HomoSeq = 'seqAll/VirusHostFinalX.fasta'
'''
Virus_Species = ["DENV","HCV","HHV4","HHV8","HIV1","InfluenzaA"]
Virus_family = ["Coronaviridae","Filoviridae","Flaviviridae","Herpesviridae","Orthomyxoviridae","Paramyxoviridae","Parvoviridae","Retroviridae"]
#Virus_Species = ["DENV"]
#Virus_family = ["Coronaviridae"]

#Virus Species Train
for Virus in Virus_Species:
    train_dir = 'dataset/train/virus/species/%s/'%Virus
    res_train = 'Final/train/virus/species/%s/'%Virus
    os.makedirs(res_train , exist_ok=True)

    # 读取UniProt ID的CSV文件
    species1_ids = []
    species2_ids = []
    index_ids = []
    VirusRecord = []
    HomoRecord = []
    with open('dataset/train/virus/species/'+  Virus +'/'+Virus+'_train.csv') as f:
        for line in f:
            columns = line.strip().split(',')
            species1_ids.append(columns[0])
            species2_ids.append(columns[1])
            index_ids.append(columns[2])

    with open(VirusSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            VirusRecord.append(record)

    with open(res_train + Virus+'_Pathogen.fasta', 'w') as f:
        for i in species1_ids:
            for j in VirusRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')

    with open(HomoSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            HomoRecord.append(record)

    with open(res_train + Virus+'_Homo.fasta', 'w') as f:
        for i in species2_ids:
            for j in HomoRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')

    with open(res_train + Virus+'_label.txt', 'w') as f:
        for i in index_ids:
            f.write(i)
            f.write("\n")

#Virus Species Test
for Virus in Virus_Species:
    test_dir = 'dataset/test/virus/species/%s/'%Virus
    res_test = 'Final/test/virus/species/%s/'%Virus
    os.makedirs(res_test , exist_ok=True)
    # 读取UniProt ID的CSV文件

    species1_ids = []
    species2_ids = []
    index_ids = []
    VirusRecord = []
    HomoRecord = []
    with open('dataset/test/virus/species/'+   Virus +'/'+Virus+'_test.csv') as f:
        for line in f:
            columns = line.strip().split(',')
            species1_ids.append(columns[0])
            species2_ids.append(columns[1])
            index_ids.append(columns[2])

    with open(VirusSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            VirusRecord.append(record)

    with open(res_test + Virus+'_Pathogen.fasta', 'w') as f:
        for i in species1_ids:
            for j in VirusRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')


    with open(HomoSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            HomoRecord.append(record)

    with open(res_test + Virus+'_Homo.fasta', 'w') as f:
        for i in species2_ids:
            for j in HomoRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')

    with open(res_test + Virus+'_label.txt', 'w') as f:
        for i in index_ids:
            f.write(i)
            f.write("\n")

#Virus Family Train
for Virus in Virus_family:
    train_dir = 'dataset/train/virus/family/%s/'%Virus
    res_train = 'Final/train/virus/family/%s/'%Virus
    os.makedirs(res_train , exist_ok=True)

    # 读取UniProt ID的CSV文件
    species1_ids = []
    species2_ids = []
    index_ids = []
    VirusRecord = []
    HomoRecord = []
    with open('dataset/train/virus/family/'+   Virus +'/'+Virus+'_train.csv') as f:
        for line in f:
            columns = line.strip().split(',')
            species1_ids.append(columns[0])
            species2_ids.append(columns[1])
            index_ids.append(columns[2])

    with open(VirusSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            VirusRecord.append(record)

    with open(res_train + Virus+'_Pathogen.fasta', 'w') as f:
        for i in species1_ids:
            for j in VirusRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')


    with open(HomoSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            HomoRecord.append(record)

    with open(res_train + Virus+'_Homo.fasta', 'w') as f:
        for i in species2_ids:
            for j in HomoRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')

    with open(res_train + Virus+'_label.txt', 'w') as f:
        for i in index_ids:
            f.write(i)
            f.write("\n")

#Virus Family Test
for Virus in Virus_family:
    test_dir = 'dataset/test/virus/family/%s/'%Virus
    res_test = 'Final/test/virus/family/%s/'%Virus
    os.makedirs(res_test , exist_ok=True)
    # 读取UniProt ID的CSV文件
    species1_ids = []
    species2_ids = []
    index_ids = []
    VirusRecord = []
    HomoRecord = []
    with open('dataset/test/virus/family/'+   Virus +'/'+Virus+'_test.csv') as f:
        for line in f:
            columns = line.strip().split(',')
            species1_ids.append(columns[0])
            species2_ids.append(columns[1])
            index_ids.append(columns[2])

    with open(VirusSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            VirusRecord.append(record)

    with open(res_test + Virus+'_Pathogen.fasta', 'w') as f:
        for i in species1_ids:
            for j in VirusRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')


    with open(HomoSeq) as f:
        for record in SeqIO.parse(f, 'fasta'):
            HomoRecord.append(record)

    with open(res_test + Virus+'_Homo.fasta', 'w') as f:
        for i in species2_ids:
            for j in HomoRecord:
                if i in j.id:
                    SeqIO.write(j, f, 'fasta')

    with open(res_test + Virus+'_label.txt', 'w') as f:
        for i in index_ids:
            f.write(i)
            f.write("\n")

'''
#####All################################
species1_ids = []
species2_ids = []
index_ids = []
VirusRecord = []
HomoRecord = []
os.makedirs('Final/train/virus/all/' , exist_ok=True)
os.makedirs('Final/test/virus/all/' , exist_ok=True)
with open('dataset/train/virus/all/All_train.csv') as f:
    for line in f:
        columns = line.strip().split(',')
        species1_ids.append(columns[0])
        species2_ids.append(columns[1])
        index_ids.append(columns[2])

with open(VirusSeq) as f:
    for record in SeqIO.parse(f, 'fasta'):
        VirusRecord.append(record)

with open('Final/train/virus/all/All_Pathogen.fasta', 'w') as f:
    for i in species1_ids:
        for j in VirusRecord:
            if i in j.id:
                SeqIO.write(j, f, 'fasta')


with open(HomoSeq) as f:
    for record in SeqIO.parse(f, 'fasta'):
        HomoRecord.append(record)

with open('Final/train/virus/all/All_Homo.fasta', 'w') as f:
    for i in species2_ids:
        for j in HomoRecord:
            if i in j.id:
                SeqIO.write(j, f, 'fasta')

with open('Final/train/virus/all/label.txt', 'w') as f:
    for i in index_ids:
        f.write(i)
        f.write("\n")

#####All Test################################
species1_ids = []
species2_ids = []
index_ids = []
VirusRecord = []
HomoRecord = []
with open('dataset/test/virus/all/All_test.csv') as f:
    for line in f:
        columns = line.strip().split(',')
        species1_ids.append(columns[0])
        species2_ids.append(columns[1])
        index_ids.append(columns[2])

with open(VirusSeq) as f:
    for record in SeqIO.parse(f, 'fasta'):
        VirusRecord.append(record)

with open('Final/test/virus/all/All_Pathogen.fasta', 'w') as f:
    for i in species1_ids:
        for j in VirusRecord:
            if i in j.id:
                SeqIO.write(j, f, 'fasta')

with open(HomoSeq) as f:
    for record in SeqIO.parse(f, 'fasta'):
        HomoRecord.append(record)

with open('Final/test/virus/all/All_Homo.fasta', 'w') as f:
    for i in species2_ids:
        for j in HomoRecord:
            if i in j.id:
                SeqIO.write(j, f, 'fasta')

with open('Final/test/virus/all/label.txt', 'w') as f:
    for i in index_ids:
        f.write(i)
        f.write("\n")
