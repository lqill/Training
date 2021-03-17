from keras import backend as K
from keras.models import Sequential
import keras.backend as K
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os.path as path
import json


# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall


# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision


# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))


# def focal_loss(y_true, y_pred):
#     gamma = 2
#     alpha = 0.25
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#     return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def make_model(node: int, layers: int, lrate: float):
    model = Sequential()
    model.add(Dense(node, input_dim=5, activation='relu'))
    for l in range(layers):
        model.add(Dense(node, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer=Adam(
        lr=lrate), metrics=['acc'])
    # model.summary()
    return model


def scale(df_column: pd.Series):
    scaler = MinMaxScaler()
    return scaler.fit_transform(df_column.values.reshape(-1, 1))


class P:
    def __init__(self, similarity: str) -> None:
        self.tipe = similarity

    def p(self, folder: str, file: str):
        return path.join(self.tipe, path.join(folder, file))


def nama(epoch: int, batch_size: int, layers: int, lr: float, node: int):
    nama_file = "epoch={}-batch_size={}-layers={}-lr={}-node={}".format(
        epoch, batch_size, layers, lr, node)
    return nama_file


def simpan_akurasi_lama(training, f):
    acc = training.history["acc"][-1]
    best_acc = max(training.history["acc"])
    val_acc = training.history["val_acc"][-1]
    loss = training.history["loss"][-1]
    val_loss = training.history["val_loss"][-1]
    f.write(
        "accuracy,{}\nmax_accuracy,{}\nval_accuracy,{}\nloss,{}\nval_loss,{}".format(
            acc, best_acc, val_acc, loss, val_loss
        ))


def simpan_akurasi(training, f):
    json_ = {}
    json_["acc"] = training.history["acc"][-1]
    json_["best_acc"] = max(training.history["acc"])
    json_["val_acc"] = training.history["val_acc"][-1]
    json_["loss"] = training.history["loss"][-1]
    json_["val_loss"] = training.history["val_loss"][-1]
    json.dump(json_, f)


def cek_arg(args):
    LIST_ARGS = [args.layer, args.node,
                 args.lr, args.batchsize, args.epoch]
    list_baru = []
    for arg in LIST_ARGS:
        if type(arg) == str:
            list_baru.append([int(arg)])
        else:
            list_baru.append(arg)
    return tuple(list_baru)
