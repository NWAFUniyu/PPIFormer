from models_onehot import get_model
import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint,EarlyStopping
from sklearn.metrics import roc_auc_score,average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers import Adadelta

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config=tf.compat.v1.ConfigProto()
sess=tf.compat.v1.Session(config=config)
config.gpu_options.allow_growth = True
#多卡训练
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置使用的GPU内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 指定使用的GPU
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
    except RuntimeError as e:
        print(e)


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
        self.en = val_data[0]
        self.pr = val_data[1]
        self.y = val_data[2]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.en,self.pr])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)
        self.model.save_weights("./model_onehot/bacteria/all/AllModel%d.h5" %epoch)
        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

name = 'All'
os.makedirs('./model_onehot/bacteria/all/', exist_ok=True)
print("training on All")
Data_dir = 'predata_onehot/train/bacteria/all/'
max_len_pr1 = 1500
max_len_pr2 = 1500
train = np.load(Data_dir + 'All_train.npz')
X_pr1_tra, X_pr2_tra, y_tra = train['X_pr1_tra'], train['X_pr2_tra'], train['y_tra']

X_pr1_tra, X_pr1_val, X_pr2_tra, X_pr2_val, y_tra, y_val = train_test_split(
    X_pr1_tra, X_pr2_tra, y_tra, test_size=0.05, stratify=y_tra, random_state=250)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = get_model(max_len_pr1, max_len_pr2)
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
    model.summary()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    back = roc_callback(val_data=[X_pr1_val, X_pr2_val, y_val], name=name)
    batch_size = 64
    train_generator = data_generator(X_pr1_tra, X_pr2_tra, y_tra, batch_size=batch_size)
        history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(X_pr1_tra) // batch_size,
        epochs=50,
        validation_data=([X_pr1_val, X_pr2_val], y_val),
        callbacks=[back,early_stop])
