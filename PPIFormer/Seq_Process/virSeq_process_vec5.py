import os

import pandas as pd
import itertools
import numpy as np

def read_matrix(matrix_file):
    matrix = {}
    with open(matrix_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            aa = parts[0]
            values = list(map(float, parts[1:]))
            matrix[aa] = values
    return matrix

def get_embedding(sequence, matrix):
    embedding = []
    for aa in sequence:
        if aa in matrix:
            embedding.append(matrix[aa])
        else:
            # If an unknown amino acid is encountered, you can handle it accordingly.
            # Here, we assign a placeholder embedding of all zeros.
            embedding.append([0.0] * len(matrix['A']))  # Using the length of the first row as a reference
    return embedding





Virus_Species = ["DENV","HCV","HHV4","HHV8","HIV1","InfluenzaA"]
Virus_family = ["Coronaviridae","Filoviridae","Flaviviridae","Herpesviridae","Orthomyxoviridae","Paramyxoviridae","Parvoviridae","Retroviridae"]
matrix_file = "vec5_CTC.txt" # Replace with the actual file path
matrix = read_matrix(matrix_file)

for virus in Virus_Species:
    #Data_dir = 'Data/train/virus/species/%s/' % virus
    #resdir = 'predata_vec5/train/virus/species/%s/' % virus
    Data_dir = 'Data/test/virus/species/%s/' % virus
    resdir = 'predata_vec5/test/virus/species/%s/' % virus
    os.makedirs(resdir, exist_ok=True)
    print ('Experiment on %s dataset' % virus)
    pr1_tra = open(Data_dir + '%s_Pathogen.fasta' % virus, 'r').read().splitlines()[1::2]
    pr1=[]

    for pr in pr1_tra:
        if len(pr) < 2000:
            pr = pr + 'X'*(2000-len(pr))
        else:
            pr = pr[:2000]
        #print(pr)
        protein_embedding = get_embedding(pr, matrix)
        pr1.append(protein_embedding)


    pr2_tra = open(Data_dir + '%s_Homo.fasta' % virus, 'r').read().splitlines()[1::2]
    pr2=[]

    for pr in pr2_tra:
        if len(pr) < 2000:
            pr = pr + 'X' * (2000 - len(pr))
        else:
            pr = pr[:2000]
        protein_embedding = get_embedding(pr, matrix)
        #print(len(protein_embedding))
        pr2.append(protein_embedding)


    y_tra = np.loadtxt(Data_dir + '%s_label.txt' % virus)
    #np.savez(resdir + '%s_train.npz' % virus, X_pr1_tra=pr1, X_pr2_tra=pr2, y_tra=y_tra)
    np.savez(resdir + '%s_test.npz'% virus, X_pr1_tes=pr1, X_pr2_tes=pr2, y_tes=y_tra)
    #print(len(pr2))


    print('pos_samples:' + str(int(sum(y_tra))))
    print('neg_samples:' + str(len(y_tra) - int(sum(y_tra))))

for virus in Virus_family:
    #Data_dir = 'Data/train/virus/family/%s/' % virus
    #resdir = 'predata_vec5/train/virus/family/%s/' % virus
    Data_dir = 'Data/test/virus/family/%s/' % virus
    resdir = 'predata_vec5/test/virus/family/%s/' % virus
    os.makedirs(resdir, exist_ok=True)
    print ('Experiment on %s dataset' % virus)
    pr1_tra = open(Data_dir + '%s_Pathogen.fasta' % virus, 'r').read().splitlines()[1::2]
    pr1=[]

    for pr in pr1_tra:
        if len(pr) < 2000:
            pr = pr + 'X'*(2000-len(pr))
        else:
            pr = pr[:2000]
        #print(pr)
        protein_embedding = get_embedding(pr, matrix)
        pr1.append(protein_embedding)


    pr2_tra = open(Data_dir + '%s_Homo.fasta' % virus, 'r').read().splitlines()[1::2]
    pr2=[]

    for pr in pr2_tra:
        if len(pr) < 2000:
            pr = pr + 'X' * (2000 - len(pr))
        else:
            pr = pr[:2000]
        protein_embedding = get_embedding(pr, matrix)
        #print(len(protein_embedding))
        pr2.append(protein_embedding)


    y_tra = np.loadtxt(Data_dir + '%s_label.txt' % virus)
    #np.savez(resdir + '%s_train.npz' % virus, X_pr1_tra=pr1, X_pr2_tra=pr2, y_tra=y_tra)
    np.savez(resdir + '%s_test.npz'% virus, X_pr1_tes=pr1, X_pr2_tes=pr2, y_tes=y_tra)
    #print(len(pr2))


    print('pos_samples:' + str(int(sum(y_tra))))
    print('neg_samples:' + str(len(y_tra) - int(sum(y_tra))))

