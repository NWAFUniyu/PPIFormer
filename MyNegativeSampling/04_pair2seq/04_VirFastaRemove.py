from Bio import SeqIO
import os
def myReadCsv(filename):
    data = []
    with open(filename,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            line = line.split(",")
            data.append(line)
    return data

# 打开fasta文件，读取所有蛋白质序列并去除注释信息
Virus_Species = ["DENV","HCV","HHV4","HHV8","HIV1","InfluenzaA"]
Virus_family = ["Coronaviridae","Filoviridae","Flaviviridae","Herpesviridae","Orthomyxoviridae","Paramyxoviridae","Parvoviridae","Retroviridae"]

#species train
for virus in Virus_Species:
    sequences = []
    train = 'Final/train/virus/species/%s/' % virus
    train_res = 'Data/train/virus/species/%s/' % virus
    os.makedirs(train_res, exist_ok=True)
    #Homo
    with open(train+virus+'_Homo.fasta', "r") as fasta_file:
        with open(train_res+virus+'_Homo.fasta', "w") as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                output_file.write(">%s\n" % record.name)
                output_file.write("%s\n" % record.seq)
    #Pathogen
    with open(train+virus+'_Pathogen.fasta', "r") as fasta_file2:
        with open(train_res+virus+'_Pathogen.fasta', "w") as output_file2:
            for record in SeqIO.parse(fasta_file2, "fasta"):
                output_file2.write(">%s\n" % record.name)
                output_file2.write("%s\n" % record.seq)

    index = myReadCsv(train+virus+'_label.txt')
    with open(train_res+virus+'_label.txt', "w") as label_file:
        flag = 0
        for i in index:
            if flag == 0:
                flag = 1
            else:
                label_file.write(str(i[0]))
                label_file.write("\n")

#species test
for virus in Virus_Species:
    sequences = []
    test = 'Final/test/virus/species/%s/' % virus
    test_res = 'Data/test/virus/species/%s/' % virus
    os.makedirs(test_res, exist_ok=True)
    #Homo
    with open(test+virus+'_Homo.fasta', "r") as fasta_file:
        with open(test_res+virus+'_Homo.fasta', "w") as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                output_file.write(">%s\n" % record.name)
                output_file.write("%s\n" % record.seq)
    #Pathogen
    with open(test+virus+'_Pathogen.fasta', "r") as fasta_file2:
        with open(test_res+virus+'_Pathogen.fasta', "w") as output_file2:
            for record in SeqIO.parse(fasta_file2, "fasta"):
                output_file2.write(">%s\n" % record.name)
                output_file2.write("%s\n" % record.seq)

    index = myReadCsv(test+virus+'_label.txt')
    with open(test_res+virus+'_label.txt', "w") as label_file:
        flag = 0
        for i in index:
            if flag == 0:
                flag = 1
            else:
                label_file.write(str(i[0]))
                label_file.write("\n")

#family train
for virus in Virus_family:
    sequences = []
    train = 'Final/train/virus/family/%s/' % virus
    train_res = 'Data/train/virus/family/%s/' % virus
    os.makedirs(train_res, exist_ok=True)
    #Homo
    with open(train+virus+'_Homo.fasta', "r") as fasta_file:
        with open(train_res+virus+'_Homo.fasta', "w") as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                output_file.write(">%s\n" % record.name)
                output_file.write("%s\n" % record.seq)
    #Pathogen
    with open(train+virus+'_Pathogen.fasta', "r") as fasta_file2:
        with open(train_res+virus+'_Pathogen.fasta', "w") as output_file2:
            for record in SeqIO.parse(fasta_file2, "fasta"):
                output_file2.write(">%s\n" % record.name)
                output_file2.write("%s\n" % record.seq)

    index = myReadCsv(train+virus+'_label.txt')
    with open(train_res+virus+'_label.txt', "w") as label_file:
        flag = 0
        for i in index:
            if flag == 0:
                flag = 1
            else:
                label_file.write(str(i[0]))
                label_file.write("\n")
#family test
for virus in Virus_family:
    sequences = []
    test = 'Final/test/virus/family/%s/' % virus
    test_res = 'Data/test/virus/family/%s/' % virus
    os.makedirs(test_res, exist_ok=True)
    #Homo
    with open(test+virus+'_Homo.fasta', "r") as fasta_file:
        with open(test_res+virus+'_Homo.fasta', "w") as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                output_file.write(">%s\n" % record.name)
                output_file.write("%s\n" % record.seq)
    #Pathogen
    with open(test+virus+'_Pathogen.fasta', "r") as fasta_file2:
        with open(test_res+virus+'_Pathogen.fasta', "w") as output_file2:
            for record in SeqIO.parse(fasta_file2, "fasta"):
                output_file2.write(">%s\n" % record.name)
                output_file2.write("%s\n" % record.seq)

    index = myReadCsv(test+virus+'_label.txt')
    with open(test_res+virus+'_label.txt', "w") as label_file:
        flag = 0
        for i in index:
            if flag == 0:
                flag = 1
            else:
                label_file.write(str(i[0]))
                label_file.write("\n")


# All train
os.makedirs("Data/train/virus/all/", exist_ok=True)
sequences = []
with open("Final/train/virus/all/All_Homo.fasta", "r") as fasta_file:
    with open("Data/train/virus/all/All_Homo.fasta", "w") as output_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            output_file.write(">%s\n" % record.name)
            output_file.write("%s\n" % record.seq)

with open("Final/train/virus/all/All_Pathogen.fasta", "r") as fasta_file2:
    with open("Data/train/virus/all/All_Pathogen.fasta", "w") as output_file2:
        for record in SeqIO.parse(fasta_file2, "fasta"):
            output_file2.write(">%s\n" % record.name)
            output_file2.write("%s\n" % record.seq)

index = myReadCsv('Final/train/virus/all/label.txt')
with open("Data/train/virus/all/All_label.txt", "w") as label_file:
    flag = 0
    for i in index:
        if flag == 0:
            flag = 1
        else:
            label_file.write(str(i[0]))
            label_file.write("\n")

# All test
os.makedirs("Data/test/virus/all/", exist_ok=True)
sequences = []
with open("Final/test/virus/all/All_Homo.fasta", "r") as fasta_file:
    with open("Data/test/virus/all/All_Homo.fasta", "w") as output_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            output_file.write(">%s\n" % record.name)
            output_file.write("%s\n" % record.seq)

with open("Final/test/virus/all/All_Pathogen.fasta", "r") as fasta_file2:
    with open("Data/test/virus/all/All_Pathogen.fasta", "w") as output_file2:
        for record in SeqIO.parse(fasta_file2, "fasta"):
            output_file2.write(">%s\n" % record.name)
            output_file2.write("%s\n" % record.seq)

index = myReadCsv('Final/test/virus/all/label.txt')
with open("Data/test/virus/all/All_label.txt", "w") as label_file:
    flag = 0
    for i in index:
        if flag == 0:
            flag = 1
        else:
            label_file.write(str(i[0]))
            label_file.write("\n")
