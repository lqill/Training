import pandas as pd
import json
from glob import glob
from library import P
from ast import literal_eval as la
p = P("jaro")
LIST_OF_FILE = glob(p.p("report_testing", "*.json"))

list_ = []
for file in LIST_OF_FILE:
    with open(file, 'r') as f:
        report = json.load(f)
    for atribut in file.split("\\")[-1].split("/")[-1].replace(".json", "").split("-"):
        report[atribut.split('=')[0]] = la(atribut.split('=')[1])
    list_.append(report)

df = pd.DataFrame(list_)
# Nama file ganti
# df.to_excel("tabel_1.0.xlsx")
df.to_excel("tabel_1.1.xlsx")
