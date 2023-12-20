#----------------------------------------------------------
# Copyright (c) 2022, Northwest A&F University
# All rights reserved.
#
# File Name：02_BacRemove.py
#
# Abstract：This file is used to remove the interactions from possible-interactions
# input：Bac_non_interactions.csv
#        BacteriaInteractionTablePH.csv
# output：Bac_updated.csv
# Version：1.0
# Author：Ni Yu
# Finished Date：2022/12/6
# Instead version：None
#-----------------------------------------------------------

import csv

# Load the first csv file into a set
with open('data/Bacteria/BacteriaInteractionTablePH.csv', 'r') as file1:
    reader1 = csv.reader(file1)
    rows1 = set(tuple(row) for row in reader1)

# Load the second csv file into a set
with open('data/possible/Bac_possible_interactions.csv', 'r') as file2, open('data/possible/Bac_update2.csv', 'w', newline='') as output:
    reader2 = csv.reader(file2)
    writer = csv.writer(output)
    for row in reader2:
        if tuple(row) not in rows1:
            writer.writerow(row)