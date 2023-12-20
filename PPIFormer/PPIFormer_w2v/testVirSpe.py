import os
import csv
from models import get_model
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from tensorflow.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config=tf.compat.v1.ConfigProto()
sess=tf.compat.v1.Session(config=config)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

def data_generator(X_pr1_tra, X_pr2_tra, batch_size=32):
    num_samples = X_pr1_tra.shape[0]
    while True:
        indices = np.random.choice(num_samples, batch_size)
        batch_X_pr1_tra = X_pr1_tra[indices]
        batch_X_pr2_tra = X_pr2_tra[indices]
        yield [batch_X_pr1_tra, batch_X_pr2_tra]

Virus_Species = ["DENV","HCV","HHV4","HHV8","HIV1","InfluenzaA"]
m = Virus_Species[0]
for name in Virus_Species:
    model = None
    max_len_en = 2000
    max_len_pr = 2000
    model = get_model(max_len_en, max_len_pr)
    modeldir = 'model/virus/species/' + m + '/' + m + 'Model.h5'
    model.load_weights(modeldir)
    Data_dir = 'predata/test/virus/species/%s/' % name
    test = np.load(Data_dir + '%s_test.npz' % name)
    X_pr1_tes, X_pr2_tes, y_tes = test['X_pr1_tes'], test['X_pr2_tes'], test['y_tes']
    print("****************Testing %s specific model on %s****************" % (m, name)) 
    aucall = 0.0
    auprall = 0.0
    for i in range(5):
    # 设置模型为测试模式
        K.set_learning_phase(0)
        y_pred=model.predict([X_pr1_tes, X_pr2_tes])    
        auc = roc_auc_score(y_tes, y_pred)
        aupr = average_precision_score(y_tes, y_pred)
        aucall = aucall + auc
        auprall = auprall +aupr
        print("AUC : ", auc)
        print("AUPR : ", aupr)
    averageAuc = aucall/5
    averageAupr = auprall/5
    print("Average AUC : ", averageAuc)
    print("Average AUPR : ", averageAupr)
