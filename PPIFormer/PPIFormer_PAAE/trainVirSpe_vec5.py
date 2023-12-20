from models_vec5 import get_model
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef,average_precision_score
import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint,EarlyStopping
from sklearn.metrics import roc_auc_score,average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config=tf.compat.v1.ConfigProto()
sess=tf.compat.v1.Session(config=config)
config.gpu_options.allow_growth = True

def data_generator(X_pr1_tra, X_pr2_tra, y_tra, batch_size=32):
    num_samples = X_pr1_tra.shape[0]
    num_batches = num_samples // batch_size
    
    # Shuffle the indices for the samples
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    while True:
        for i in range(num_batches):
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            batch_X_pr1_tra = X_pr1_tra[batch_indices]
            batch_X_pr2_tra = X_pr2_tra[batch_indices]
            batch_y_tra = y_tra[batch_indices]
            yield [batch_X_pr1_tra, batch_X_pr2_tra], batch_y_tra

class roc_callback(Callback):
    def __init__(self, val_data,name):
        self.pr1 = val_data[0]
        self.pr2 = val_data[1]
        self.y = val_data[2]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.pr1,self.pr2])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)
        # other
        y_pred_label = np.zeros(len(y_pred))
        y_pred_label[np.where(y_pred.flatten() >= 0.5)] = 1
        y_pred_label[np.where(y_pred.flatten() < 0.5)] = 0
        y_true = self.y
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
        recall = recall_score(y_true, y_pred_label)
        spec = tn / (tn + fp)
        npv = tn / (tn + fn)
        acc = accuracy_score(y_true, y_pred_label)
        prec = precision_score(y_true, y_pred_label)
        mcc = matthews_corrcoef(y_true, y_pred_label)
        f1 = 2 * prec * recall / (prec + recall)
        print("Sensitivity: %.4f, Specificity: %.4f, Accuracy: %.4f, PPV: %.4f, NPV: %.4f, MCC: %.4f,F1: %.4f" \
             % (recall * 100,     spec * 100,        acc * 100,      prec * 100, npv * 100, mcc*100, f1 * 100))
        self.model.save_weights("./model_vec5/virus/species/%s/%sModel%d.h5" % (self.name, self.name, epoch))
        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


Virus_Species = ["DENV","HCV","HHV4","HHV8","HIV1","InfluenzaA"]

for name in Virus_Species:
    os.makedirs('./model_vec5/virus/species/%s/' % name, exist_ok=True)
    print("training on " + name)
    Data_dir = 'predata_vec5/train/virus/species/%s/' % name

    max_len_pr1 = 2000
    max_len_pr2 = 2000
    emb_dim = 100

    train = np.load(Data_dir + '%s_train.npz' % name)
    X_pr1_tra, X_pr2_tra, y_tra = train['X_pr1_tra'], train['X_pr2_tra'], train['y_tra']

    X_pr1_tra, X_pr1_val, X_pr2_tra, X_pr2_val, y_tra, y_val = train_test_split(
        X_pr1_tra, X_pr2_tra, y_tra, test_size=0.05, stratify=y_tra, random_state=250)
 
    model = get_model(max_len_pr1, max_len_pr2)
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
    model.summary()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    back = roc_callback(val_data=[X_pr1_val, X_pr2_val, y_val], name=name)
    batch_size = 32
    train_generator = data_generator(X_pr1_tra, X_pr2_tra, y_tra, batch_size=batch_size)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(X_pr1_tra) // batch_size,
        epochs=50,
        validation_data=([X_pr1_val, X_pr2_val], y_val),
        callbacks=[back,early_stop]
    )
