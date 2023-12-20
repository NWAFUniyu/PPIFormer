#from seq2tensor import s2t
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.utils import multi_gpu_model, Sequence, np_utils
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef,average_precision_score
from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, add
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import Concatenate, concatenate, subtract, multiply
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, \
    matthews_corrcoef, average_precision_score
import sys
from numpy import linalg as LA
import scipy
import numpy as np
import os
import tensorflow as tf
#from utils import *
from models import *
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.python.keras.layers import GRU, LSTM
from tensorflow.python.keras.layers import CuDNNGRU, CuDNNLSTM
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
        #

        self.model.save_weights("./model/pioneer_vec5/rcnn/%s/%sModel%d.h5" % (self.name, self.name, epoch))
        #self.model.save_weights("./model/virus/species/retrain/%s/%sModel%d.h5" % (self.name, self.name, epoch))
        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def data_generator(X_pr1_tra, X_pr2_tra, y_tra, batch_size):
    num_samples = X_pr1_tra.shape[0]
    num_batches = num_samples // batch_size

    # Shuffle the indices for the samples
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    while True:
        for i in range(num_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch_X_pr1_tra = X_pr1_tra[batch_indices]
            batch_X_pr2_tra = X_pr2_tra[batch_indices]
            batch_y_tra = y_tra[batch_indices]
            yield [batch_X_pr1_tra, batch_X_pr2_tra], batch_y_tra

def build_model():
    seq_input1 = Input(shape=(seq_size, 13), name='seq1')
    seq_input2 = Input(shape=(seq_size, 13), name='seq2')
    l1 = Conv1D(hidden_dim, 3)
    c1 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)
    r1 = Bidirectional(c1)

    l2 = Conv1D(hidden_dim, 3)
    c2 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)
    r2 = Bidirectional(c2)

    l3 = Conv1D(hidden_dim, 3)
    c3 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)
    r3 = Bidirectional(c3)

    l4 = Conv1D(hidden_dim, 3)
    c4 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)
    r4 = Bidirectional(c4)

    l5 = Conv1D(hidden_dim, 3)
    c5 = tf.keras.layers.GRU(hidden_dim, return_sequences=True)
    r5 = Bidirectional(c5)

    l6 = Conv1D(hidden_dim, 3)
    """
    l1 = Conv1D(hidden_dim, 3)
    r1 = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l2 = Conv1D(hidden_dim, 3)
    r2 = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l3 = Conv1D(hidden_dim, 3)
    r3 = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l4 = Conv1D(hidden_dim, 3)
    r4 = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l5 = Conv1D(hidden_dim, 3)
    r5 = Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l6 = Conv1D(hidden_dim, 3)
    """
    s1 = MaxPooling1D(3)(l1(seq_input1))
    s1 = concatenate([r1(s1), s1])
    s1 = MaxPooling1D(3)(l2(s1))
    s1 = concatenate([r2(s1), s1])
    s1 = MaxPooling1D(3)(l3(s1))
    s1 = concatenate([r3(s1), s1])
    s1 = MaxPooling1D(3)(l4(s1))
    s1 = concatenate([r4(s1), s1])
    s1 = MaxPooling1D(3)(l5(s1))
    s1 = concatenate([r5(s1), s1])
    s1 = l6(s1)
    s1 = GlobalAveragePooling1D()(s1)
    
    s2 = MaxPooling1D(3)(l1(seq_input2))
    s2 = concatenate([r1(s2), s2])
    s2 = MaxPooling1D(3)(l2(s2))
    s2 = concatenate([r2(s2), s2])
    s2 = MaxPooling1D(3)(l3(s2))
    s2 = concatenate([r3(s2), s2])
    s2 = MaxPooling1D(3)(l4(s2))
    s2 = concatenate([r4(s2), s2])
    s2 = MaxPooling1D(3)(l5(s2))
    s2 = concatenate([r5(s2), s2])
    s2 = l6(s2)
    s2 = GlobalAveragePooling1D()(s2)
    merge_text = multiply([s1, s2])
    x = Dense(100, activation='linear')(merge_text)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim + 7) / 2), activation='linear')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(1, activation='sigmoid')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model


# weights_file = f'model_rcnn_denovo.h5'

seq_size = 2000
MAXLEN = seq_size
#seq2t = s2t('vec5_CTC.txt')
hidden_dim = 50
#dim = seq2t.dim
epochs = 30
num_gpus = 1
batch_size = 32
steps = 500
verbose = 1
virus_fasta = "data/VirusPathogenFinalX.fasta"
human_fasta = "data/VirusHostFinalX.fasta"

Virus_Species = ["DENV", "HCV", "HHV4",  "HHV8", "HIV1", "InfluenzaA", "ZIKV"]


for name in Virus_Species:
    os.makedirs('./model/pioneer_vec5/rcnn/' + name, exist_ok=True)
    print("training on " + name)
    Data_dir = 'predata_vec5/train/virus/species/%s/' % name

    max_len_pr1 = 2000
    max_len_pr2 = 2000
    emb_dim = 100
    train = np.load(Data_dir + '%s_train.npz' % name)
    X_pr1_tra, X_pr2_tra, y_tra = train['X_pr1_tra'], train['X_pr2_tra'], train['y_tra']

    X_pr1_tra, X_pr1_val, X_pr2_tra, X_pr2_val, y_tra, y_val = train_test_split(
        X_pr1_tra, X_pr2_tra, y_tra, test_size=0.1, stratify=y_tra, random_state=2023)

    model = build_model()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    back = roc_callback(val_data=[X_pr1_val, X_pr2_val, y_val], name=name)

    batch_size = 32
    train_generator = data_generator(X_pr1_tra, X_pr2_tra, y_tra, batch_size=batch_size)
    val_generator = data_generator(X_pr1_val, X_pr2_val, y_val, batch_size=batch_size)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(X_pr1_tra) // batch_size,
        epochs=30,
        validation_data=val_generator,
        validation_steps=len(X_pr1_val) // batch_size,
        callbacks=[back, early_stop]
    )


