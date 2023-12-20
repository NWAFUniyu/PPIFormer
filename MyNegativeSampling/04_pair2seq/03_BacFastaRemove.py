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
'''
# 打开fasta文件，读取所有蛋白质序列并去除注释信息
Bacteria_Species = ["Bacillus anthracis","Burkholderia mallei","Francisella tularensis","Yersinia pestis","Others"]

#species train
for Bacteria in Bacteria_Species:
    sequences = []
    train = 'Final/train/bacteria/species/%s/' % Bacteria
    train_res = 'Data/train/bacteria/species/%s/' % Bacteria
    os.makedirs(train_res, exist_ok=True)
    #Homo
    with open(train+Bacteria+'_Homo.fasta', "r") as fasta_file:
        with open(train_res+Bacteria+'_Homo.fasta', "w") as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                output_file.write(">%s\n" % record.name)
                output_file.write("%s\n" % record.seq)
    #Pathogen
    with open(train+Bacteria+'_Pathogen.fasta', "r") as fasta_file2:
        with open(train_res+Bacteria+'_Pathogen.fasta', "w") as output_file2:
            for record in SeqIO.parse(fasta_file2, "fasta"):
                output_file2.write(">%s\n" % record.name)
                output_file2.write("%s\n" % record.seq)

    index = myReadCsv(train+Bacteria+'_label.txt')
    with open(train_res+Bacteria+'_label.txt', "w") as label_file:
        flag = 0
        for i in index:
            if flag == 0:
                flag = 1
            else:
                label_file.write(str(i[0]))
                label_file.write("\n")

#species test
for Bacteria in Bacteria_Species:
    sequences = []
    test = 'Final/test/bacteria/species/%s/' % Bacteria
    test_res = 'Data/test/bacteria/species/%s/' % Bacteria
    os.makedirs(test_res, exist_ok=True)
    #Homo
    with open(test+Bacteria+'_Homo.fasta', "r") as fasta_file:
        with open(test_res+Bacteria+'_Homo.fasta', "w") as output_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                output_file.write(">%s\n" % record.name)
                output_file.write("%s\n" % record.seq)
    #Pathogen
    with open(test+Bacteria+'_Pathogen.fasta', "r") as fasta_file2:
        with open(test_res+Bacteria+'_Pathogen.fasta', "w") as output_file2:
            for record in SeqIO.parse(fasta_file2, "fasta"):
                output_file2.write(">%s\n" % record.name)
                output_file2.write("%s\n" % record.seq)

    index = myReadCsv(test+Bacteria+'_label.txt')
    with open(test_res+Bacteria+'_label.txt', "w") as label_file:
        flag = 0
        for i in index:
            if flag == 0:
                flag = 1
            else:
                label_file.write(str(i[0]))
                label_file.write("\n")

'''
# All train
os.makedirs("Data/train/bacteria/all/", exist_ok=True)
sequences = []
with open("Final/train/bacteria/all/All_Homo.fasta", "r") as fasta_file:
    with open("Data/train/bacteria/all/All_Homo.fasta", "w") as output_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            output_file.write(">%s\n" % record.name)
            output_file.write("%s\n" % record.seq)

with open("Final/train/bacteria/all/All_Pathogen.fasta", "r") as fasta_file2:
    with open("Data/train/bacteria/all/All_Pathogen.fasta", "w") as output_file2:
        for record in SeqIO.parse(fasta_file2, "fasta"):
            output_file2.write(">%s\n" % record.name)
            output_file2.write("%s\n" % record.seq)

index = myReadCsv('Final/train/bacteria/all/All_label.txt')
with open("Data/train/bacteria/all/All_label.txt", "w") as label_file:
    flag = 0
    for i in index:
        if flag == 0:
            flag = 1
        else:
            label_file.write(str(i[0]))
            label_file.write("\n")

# All test
os.makedirs("Data/test/bacteria/all/", exist_ok=True)
sequences = []
with open("Final/test/bacteria/all/All_Homo.fasta", "r") as fasta_file:
    with open("Data/test/bacteria/all/All_Homo.fasta", "w") as output_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            output_file.write(">%s\n" % record.name)
            output_file.write("%s\n" % record.seq)

with open("Final/test/bacteria/all/All_Pathogen.fasta", "r") as fasta_file2:
    with open("Data/test/bacteria/all/All_Pathogen.fasta", "w") as output_file2:
        for record in SeqIO.parse(fasta_file2, "fasta"):
            output_file2.write(">%s\n" % record.name)
            output_file2.write("%s\n" % record.seq)

index = myReadCsv('Final/test/bacteria/all/All_label.txt')
with open("Data/test/bacteria/all/All_label.txt", "w") as label_file:
    flag = 0
    for i in index:
        if flag == 0:
            flag = 1
        else:
            label_file.write(str(i[0]))
            label_file.write("\n")