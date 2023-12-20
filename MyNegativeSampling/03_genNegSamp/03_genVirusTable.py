#----------------------------------------------------------
# Copyright (c) 2022, Northwest A&F University
# All rights reserved.
#
# File Name：03_genVirusTable.py
#
# Abstract：This file is used to generate possible interaction pairs for Virus
# input：
#         VirusInteractionTable.csv: The interaction pairs of Virus
#         VirusPathogenDistance30_normalized.mat: The similarity matrix between virus Proteins
#         VirusHostDistance30_normalized.mat: The similarity matrix between host Proteins
# output：Vir_possible_interactions.csv
# Version：2.0
# Author：Ni Yu
# Finished Date：2023/05/03
# Instead version：None
#-----------------------------------------------------------
import csv
import numpy as np
import scipy.io

interactions = []
with open('data/Virus/VirusInteractionTable.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        interactions.append(row)

# 加载两个MAT文件
pathogen_similarity = scipy.io.loadmat('data/Virus/VirusPathogenDistance30.mat')['Weights']
host_similarity = scipy.io.loadmat('data/Virus/VirusHostDistance30.mat')['Weights']

# 初始化新的CSV文件
possible_interactions = []
possible_interactions.append(['Pathogen Protein ID','Host Protein ID'])

for interaction in interactions:
    h1 = int(interaction[0])
    p1 = int(interaction[1])
    # 找到p1的相似度
    p1_similarity = pathogen_similarity[p1 - 1]
    # 计算与p1相似度的阈值
    threshold = np.percentile(p1_similarity, 2.5)
    for p2 in range(len(pathogen_similarity)):
        if p2 != p1-1 and pathogen_similarity[p1-1,p2] <threshold:
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



# 写入新的CSV文件
with open('data/possible/Vir_possible_interactions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(possible_interactions)