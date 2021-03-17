import pandas as pd
import numpy as np
from library import make_model, P, nama, simpan_akurasi, cek_arg
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--layer", default=np.arange(1, 6))
parser.add_argument("--node", default=[50, 100, 150])
parser.add_argument("--lr", default=[1/(10**i) for i in range(1, 5)])
parser.add_argument("--batchsize", default=[16, 32, 64])
parser.add_argument("--epoch", default=[50, 100, 200])
args = parser.parse_args()
LIST_LAYER, LIST_NODE, LIST_LR, LIST_BATCH_SIZE, LIST_EPOCH = cek_arg(args)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

p = P("jaro")
X = pd.read_pickle(p.p("train_test", "X.pkl"))
y = pd.read_pickle(p.p("train_test", "y.pkl"))
# X_test = pd.read_pickle(Path.joinpath(FOLDER, "X_test.pkl"))
# y_test = pd.read_pickle(Path.joinpath(FOLDER, "y_test.pkl"))

# LIST_LAYER = np.arange(1, 6)  # 1 sampai 5 layer
# #LIST_NODE = [50, 100, 150]  # 50, 100, dan 150 Node dalam satu layer
# LIST_NODE=[50] # Sikok dulu bae
# LIST_LR = [1/(10**i) for i in range(1, 5)]  # 1E-01 sampai 1E-04
# LIST_BATCH_SIZE = [16, 32, 64]  # batch_size
# LIST_EPOCH =[50, 100, 200]  # epoch

for epoch in LIST_EPOCH:
    for batch_size in LIST_BATCH_SIZE:
        for layers in LIST_LAYER:
            for node in LIST_NODE:
                for lr in LIST_LR:
                    nama_file = nama(epoch, batch_size, layers, lr, node)
                    # --------------------------------------------
                    # Kalau sudah pernah training, skip
                    if os.path.isfile(p.p("history", nama_file)):
                        print("SUDAH PERNAH TRAINING : "+nama_file, "red")
                        continue
                    # --------------------------------------------
                    for i in range(10):
                        print("MODEL TUNING : "+nama_file, "red")
                    # --------------------------------------------
                    # Buat model berdasarkan tuning
                    model = make_model(node, layers, lr)
                    # --------------------------------------------
                    # fit train model
                    training = model.fit(
                        X, y, validation_split=0.2, epochs=epoch, batch_size=batch_size)
                    # --------------------------------------------
                    # simpan history
                    with open(p.p("history", nama_file), 'wb') as f:
                        pickle.dump(training.history, f)
                    # --------------------------------------------
                    # simpan akurasi dan loss sementara, belum confusion matrix
                    with open(p.p("hasil", nama_file+".json"), 'w') as f:
                        simpan_akurasi(training,f)
                    # with open(p.p("hasil", nama_file+".txt"), 'w') as f:
                    #     simpan_akurasi_lama(training, f)
                    
                    # --------------------------------------------
                    # simpan model untuk testing nanti
                    model.save(
                        p.p("model", nama_file+".pkl"))
                    # load model pakek ini
                    # model = load_model("model/{}-{}-{}.history".format(layers, lr, node))
