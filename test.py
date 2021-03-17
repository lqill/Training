import pandas as pd
import os
from keras.models import load_model
from glob import glob
from library import P
import numpy as np
import tensorflow as tf
import json
from sklearn.metrics import confusion_matrix

tf.config.experimental.set_visible_devices([], 'GPU')
p = P("jaro")

X_test = pd.read_pickle(p.p("train_test", "X_test.pkl")).values
y_test = pd.read_pickle(p.p("train_test", "y_test.pkl")).values

LIST_MODEL = glob(p.tipe+"/model/*")

for model_simpanan in LIST_MODEL:
    if os.path.isfile(p.p("report_testing", model_simpanan.split("/")[-1].split("\\")[-1].replace(".pkl", ".json"))):
        print("\n".join(["SUDAH "+model_simpanan for i in range(3)]))
        continue
    print("\n".join(["TESTING "+model_simpanan for i in range(3)]))
    model = load_model(model_simpanan)
    # y_pred = model.predict_classes(X_test)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[0, 0]
    tn = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    json_ = {"cm": cm.tolist()}
    json_["recall"] = tp / (tp + fn)
    json_["presisi"] = tp / (tp + fp)
    try:
        json_["spesifity"] = tn / (tn + fp)
    except:
        json_["spesifity"] = np.nan
    json_["akurasi"] = (tp+tn) / (tp+fp+fn+tn)
    json_["error"] = (fp + fn) / (fp + fn + tn + tp)
    json_["f1"] = (2 * json_["presisi"] * json_["recall"]) / \
        (json_["recall"] + json_["presisi"])
    with open(p.p("report_testing", model_simpanan.split("/")[-1].split("\\")[-1].replace(".pkl", ".json")), "w") as f:
        json.dump(json_, f)
    # with open(p.p("report_testing", model_simpanan.split("/")[-1].split("\\")[-1].replace(".pkl", ".csv")), "w") as f:
    #     print("cm:{}\nrecall:{}\npresisi:{}\nspesifity:{}\nakurasi:{}\nerror:{}\nf1:{}\n".format(
    #         [[tp, fp], [fn, tn]], recall, presisi, spesifity, akurasi, error, f1))
    #     f.write("cm:{}\nrecall:{}\npresisi:{}\nspesifity:{}\nakurasi:{}\nerror:{}\nf1:{}\n".format(
    #         [[tp, fp], [fn, tn]], recall, presisi, spesifity, akurasi, error, f1
    #     ))
    with open(p.p("label_testing", model_simpanan.split("/")[-1].split("\\")[-1]), "wb") as f:
        f.write(y_pred)
