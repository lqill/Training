import pandas as pd

df = pd.read_excel("tabel_1.0.xlsx", index_col=0, dtype='str')
df = df[["layers", "batch_size", "node", "lr", "akurasi"]]
for kolom in df.columns:
    if kolom != "akurasi" and kolom != "lr":
        df[kolom] = df[kolom].astype("int")
df.akurasi = df.akurasi.astype("float64")
df.lr = df.lr.astype("float64")
df = df[df.layers < 4]
cek = df.pivot_table(columns="lr", index=["layers", "batch_size", "node"], values="akurasi")

cek.to_excel("tabel_siap1.0.xlsx")
