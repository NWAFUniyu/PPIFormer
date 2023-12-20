#----------------------------------------------------------
# Copyright (c) 2022, Northwest A&F University
# All rights reserved.
#
# File Name：01_genBacTable.py
#
# Abstract：This file is used to generate possible interaction pairs for Bacteria
# input：
#       BacteriaInteractionTable.csv
#       BacteriaPathogenDistance30_normalized.mat
#       BacteriaHostDistance30_normalized.mat
# output：
#       Bac_posssible_interactions.csv
# Version：2.0
# Author：Ni Yu
# Finished Date：2023/05/03
# Instead version：None
#-----------------------------------------------------------

import csv
import numpy as np
import scipy.io

# load interactions
interactions = []
with open('data/Bacteria/BacteriaInteractionTable.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        interactions.append(row)


# load Mat
#修改日期：2021/05/03
#修改内容：改为取最小的阈值

pathogen_similarity = scipy.io.loadmat('data/Bacteria/BacteriaPathogenDistance30.mat')['Weights']
host_similarity = scipy.io.loadmat('data/Bacteria/BacteriaHostDistance30.mat')['Weights']


# initialize new CSV file
possible_interactions = []
possible_interactions.append(['Pathogen Protein ID','Host Protein ID'])

for interaction in interactions:
    h1 = int(interaction[0])
    p1 = int(interaction[1])
    #找到p1的相似度
    p1_similarity = pathogen_similarity[p1-1]
    # 计算与p1相似度的阈值
    threshold = np.percentile(p1_similarity, 2.5)
    #print(len(pathogen_similarity[p1-1]))
    for p2 in range(len(pathogen_similarity[p1-1])):
        #print(p2)
        if p2 != p1-1 and pathogen_similarity[p1-1,p2] < threshold:
            # if [str(p2 + 1), str(h1)] not in possible_interactions:
            possible_interactions.append([str(p2+1), str(h1)])
            print(len(possible_interactions))


for interaction in interactions:
    h1 = int(interaction[0])
    p1 = int(interaction[1])

    # 找到h1的相似度
    h1_similarity = host_similarity[h1 - 1]
    # 计算与p1相似度的阈值
    threshold = np.percentile(h1_similarity, 2.5)

    for h2 in range(len(host_similarity)):
        if h2 != h1-1 and host_similarity[h1-1,h2] < threshold:
            # if [str(p1), str(h2+1)] not in possible_interactions:
            possible_interactions.append([str(p1), str(h2+1)])
            print(len(possible_interactions))


#  write to CSV file
with open('data/possible/Bac_possible_interactions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(possible_interactions)


"""
for interaction in interactions:
    h1 = int(interaction[0])
    p1 = int(interaction[1])
    for p2 in range(len(pathogen_similarity)):
        if p2 != p1-1 and pathogen_similarity[p2, p1-1] < -1.96:
            # if [str(p2 + 1), str(h1)] not in possible_interactions:
            possible_interactions.append([str(p2+1), str(h1)])
            print(len(possible_interactions))

for interaction in interactions:
    h1 = int(interaction[0])
    p1 = int(interaction[1])
    for h2 in range(len(host_similarity)):
        if h2 != h1-1 and host_similarity[h2, h1-1] < -1.96:
            # if [str(p1), str(h2+1)] not in possible_interactions:
            possible_interactions.append([str(p2+1), str(h1)])
            print(len(possible_interactions))
"""