import os
from model_w2vRcnn import build_model
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config=tf.compat.v1.ConfigProto()
sess=tf.compat.v1.Session(config=config)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True


#def data_generator(X_pr1_tra, X_pr2_tra, batch_size=32):
#    num_samples = X_pr1_tra.shape[0]
#    while True:
#        indices = np.random.choice(num_samples, batch_size)
#        batch_X_pr1_tra = X_pr1_tra[indices]
#        batch_X_pr2_tra = X_pr2_tra[indices]
#        yield [batch_X_pr1_tra, batch_X_pr2_tra]

#Virus_Species = ["DENV","HCV","HHV4","HHV8","HIV1","InfluenzaA","ZIKV"]
#Virus_Species = ["HIV1","InfluenzaA","ZIKV"]
Virus_Species = ["InfluenzaA"]

#for name in Virus_Species:
m = Virus_Species[0]
name = Virus_Species[0]
model = None
max_len_en = 2000
max_len_pr = 2000
model = build_model()
modeldir = './model/pioneer/rcnn/' + m + '/' + m + 'Model6.h5'
model.load_weights(modeldir)
Data_dir = 'predata/test/virus/species/%s/' % name
test = np.load(Data_dir + '%s_test.npz' % name)
X_pr1_tes, X_pr2_tes, y_tes = test['X_pr1_tes'], test['X_pr2_tes'], test['y_tes']
print("****************Testing %s specific model on %s****************" % (m, name))
#aucall = 0.0
#auprall = 0.0
#for i in range(5):
y_pred=model.predict([X_pr1_tes, X_pr2_tes])
#threshold = 0.5  # 阈值
#y_pred = np.where(y_pred >= threshold, 1, 0)  # 将大于等于阈值的结果设为1，小于阈值的结果设为0
auc = roc_auc_score(y_tes, y_pred)
aupr = average_precision_score(y_tes, y_pred)
#auc = roc_auc_score(y_tes, y_pred_class)
#aupr = average_precision_score(y_tes, y_pred_class)
#aucall = aucall + auc
#auprall = auprall +aupr
print("AUC : ", auc)
print("AUPR : ", aupr)
#averageAuc = aucall/5
#averageAupr = auprall/5
#print("Average AUC : ", averageAuc)
#print("Average AUPR : ", averageAupr)

