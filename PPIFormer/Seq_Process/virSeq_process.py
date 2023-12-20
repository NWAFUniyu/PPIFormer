import os
from gensim.models import Word2Vec
import pandas as pd
import itertools
import numpy as np
def seq_to_kmers(seq, k=3):
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
        k (int), default 3
    Returns:
        List containing a list of kmers.
    """
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]


class Corpus(object):
    """ An iteratable for training seq2vec models. """

    def __init__(self, dir, ngram):
        self.df = pd.read_csv(dir)
        self.ngram = ngram

    def __iter__(self):
        for sentence in self.df.Seq.values:
            yield seq_to_kmers(sentence, self.ngram)


def get_protein_embedding(model,protein):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((len(protein), 100))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
#    vec = vec.astype(np.float32)
    return vec

def get_pr1_embedding(model,protein):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((2000, 100))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
#    vec = vec.astype(np.float32)
    return vec

def get_pr2_embedding(model,protein):
    """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    vec = np.zeros((2000, 100))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
    return vec

Virus_Species = ["DENV","HCV","HHV4","HHV8","HIV1","InfluenzaA"]
Virus_family = ["Coronaviridae","Filoviridae","Flaviviridae","Herpesviridae","Orthomyxoviridae","Paramyxoviridae","Parvoviridae","Retroviridae"]

model_virus = Word2Vec.load('word2vec_Model/virus_30.model')
model_homo = Word2Vec.load('word2vec_Model/human_30.model')

for virus in Virus_Species:
    #Data_dir = 'Data/train/virus/species/%s/' % virus
    #resdir = 'predata/train/virus/species/%s/' % virus
    Data_dir = 'Data/test/virus/species/%s/' % virus
    resdir = 'predata/test/virus/species/%s/' % virus
    os.makedirs(resdir, exist_ok=True)
    print ('Experiment on %s dataset' % virus)
    pr1_tra = open(Data_dir + '%s_Pathogen.fasta' % virus, 'r').read().splitlines()[1::2]
    pr1=[]

    for pr in pr1_tra:
        pr = pr[:2000]
        #print(pr)
        protein_embedding = get_pr1_embedding(model_virus, seq_to_kmers(pr))
        pr1.append(protein_embedding)


    pr2_tra = open(Data_dir + '%s_Homo.fasta' % virus, 'r').read().splitlines()[1::2]
    pr2=[]

    for pr in pr2_tra:
        pr = pr[:2000]
        protein_embedding = get_pr2_embedding(model_homo, seq_to_kmers(pr))
        pr2.append(protein_embedding)


    y_tra = np.loadtxt(Data_dir + '%s_label.txt' % virus)
    #np.savez(resdir + '%s_train.npz' % virus, X_pr1_tra=pr1, X_pr2_tra=pr2, y_tra=y_tra)
    np.savez(resdir + '%s_test.npz'% virus, X_pr1_tes=pr1, X_pr2_tes=pr2, y_tes=y_tra)
    #print(len(pr2))


    print('pos_samples:' + str(int(sum(y_tra))))
    print('neg_samples:' + str(len(y_tra) - int(sum(y_tra))))

for virus in Virus_family:
    #Data_dir = 'Data/train/virus/family/%s/' % virus
    #resdir = 'predata/train/virus/family/%s/' % virus
    Data_dir = 'Data/test/virus/family/%s/' % virus
    resdir = 'predata/test/virus/family/%s/' % virus
    os.makedirs(resdir, exist_ok=True)
    print ('Experiment on %s dataset' % virus)
    pr1_tra = open(Data_dir + '%s_Pathogen.fasta' % virus, 'r').read().splitlines()[1::2]
    pr1=[]

    for pr in pr1_tra:
        pr = pr[:2000]
        #print(pr)
        protein_embedding = get_pr1_embedding(model_virus, seq_to_kmers(pr))
        pr1.append(protein_embedding)


    pr2_tra = open(Data_dir + '%s_Homo.fasta' % virus, 'r').read().splitlines()[1::2]
    pr2=[]

    for pr in pr2_tra:
        pr = pr[:2000]
        protein_embedding = get_pr2_embedding(model_homo, seq_to_kmers(pr))
        pr2.append(protein_embedding)


    y_tra = np.loadtxt(Data_dir + '%s_label.txt' % virus)
    #np.savez(resdir + '%s_train.npz' % virus, X_pr1_tra=pr1, X_pr2_tra=pr2, y_tra=y_tra)
    np.savez(resdir + '%s_test.npz'% virus, X_pr1_tes=pr1, X_pr2_tes=pr2, y_tes=y_tra)
    #print(len(pr2))


    print('pos_samples:' + str(int(sum(y_tra))))
    print('neg_samples:' + str(len(y_tra) - int(sum(y_tra))))

