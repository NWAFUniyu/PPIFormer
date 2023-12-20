#----------------------------------------------------------
# Copyright (c) 2022, Northwest A&F University
# All rights reserved.
#
# File Name：04_VirusRemove.py
#
# Abstract：This file is used to remove the interactions from possible-interactions
# input：Vir_non_interactions.csv
#        VirusInteractionTablePH.csv
# output：Vir_updated.csv
# Version：1.0
# Author：Ni Yu
# Finished Date：2022/12/6
# Instead version：None
#-----------------------------------------------------------
import csv

# 读取第一个csv文件中的所有行
with open('data/Virus/VirusInteractionTablePH.csv', 'r') as file1:
    reader1 = csv.reader(file1)
    rows1 = set(tuple(row) for row in reader1)

# 读取第二个csv文件，将所有不在第一个csv文件中的行写入新文件
with open('data/possible/Vir_possible_interactions.csv', 'r') as file2, open('data/possible/Vir_update2.csv', 'w', newline='') as output:
    reader2 = csv.reader(file2)
    writer = csv.writer(output)
    for row in reader2:
        if tuple(row) not in rows1:
            writer.writerow(row)