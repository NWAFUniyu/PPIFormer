from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np
import tensorflow as tf
from transfomer import Transformer

def get_model(max_len_pr1, max_len_pr2):
    #input
    pr1 = Input(shape=(max_len_pr1,100,))
    pr2 = Input(shape=(max_len_pr2,100,))

    #pr1
    pr1_conv_layer = Conv1D(filters=64,#64
                                    kernel_size=40,#40
                                    padding="valid",
                                    activation='relu'
                                    )(pr1)
    pr1_max_pool_layer = MaxPooling1D(pool_size=5, strides=5)(pr1_conv_layer)
    pr1_trf = Transformer( encoder_stack=4,
                                feed_forward_size=256,
                                n_heads=8,
                                model_dim=64)(pr1_max_pool_layer)
    pr1_res = Add()([pr1_max_pool_layer, pr1_trf])  # Residual connection
    pr1_conv_layer2 = Conv1D(filters=64,#64
                                    kernel_size=40,#40
                                    padding="valid",
                                    activation='relu'
                                    )(pr1_res)

    pr1_maxpool = GlobalMaxPooling1D()(pr1_conv_layer2)
    pr1_avgpool = GlobalAveragePooling1D()(pr1_conv_layer2)
    pr1_pool = tf.concat([pr1_avgpool, pr1_maxpool], -1)


    #pr2
    pr2_conv_layer = Conv1D(filters=64,#64
                                    kernel_size=40,#40
                                    padding="valid",
                                    activation='relu'
                                    )(pr2)
    pr2_max_pool_layer = MaxPooling1D(pool_size=5, strides=5)(pr2_conv_layer)
    pr2_trf = Transformer( encoder_stack=4,
                                feed_forward_size=256,
                                n_heads=8,
                                model_dim=64)(pr2_max_pool_layer)
    pr2_res = Add()([pr2_max_pool_layer, pr2_trf])  # Residual connection
    pr2_conv_layer2 = Conv1D(filters=64,#64
                                    kernel_size=40,#40
                                    padding="valid",
                                    activation='relu'
                                    )(pr2_res)

    pr2_maxpool = GlobalMaxPooling1D()(pr2_conv_layer2)
    pr2_avgpool = GlobalAveragePooling1D()(pr2_conv_layer2)
    pr2_pool = tf.concat([pr2_avgpool, pr2_maxpool], -1)


    #concat
    subtract = Subtract()([pr1_pool, pr2_pool])
    multiply = Multiply()([pr1_pool, pr2_pool])
    merge = Concatenate(axis=1)([pr1_pool, pr2_pool, subtract, multiply])

    # my
    # bn=BatchNormalization()(merge)
    # do=Dropout(0.5)(merge)
    merge2 = Dense(100, activation='relu')(merge)
    merge3 = Dense(50, activation='relu')(merge2)
    preds = Dense(1, activation='sigmoid')(merge3)
    model = Model([pr1, pr2], preds)

    return model

