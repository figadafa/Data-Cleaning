import pandas as pd
import re

# Muat file CSV mentah
df = pd.read_csv("cleaned_kuliah.csv")

# Ambil hanya kolom teks tweet dan bersihkan
df_clean = df[['full_text']].rename(columns={'full_text': 'tweet'})
df_clean.drop_duplicates(subset='tweet', inplace=True)

# Fungsi bersihkan tweet
def bersihkan(teks):
    teks = re.sub(r"http\S+", "", teks)  # hapus link
    teks = re.sub(r"@\w+", "", teks)     # hapus mention
    teks = re.sub(r"#\w+", "", teks)     # hapus hashtag
    teks = teks.replace('\n', ' ')       # hapus newline
    return teks.strip()

df_clean['tweet'] = df_clean['tweet'].astype(str).apply(bersihkan)

# Daftar keyword stres dari user + sistem
keywords = [
    "tugas", "ujian", "skripsi", "capek", "lelah", "pusing", "burnout",
    "malas", "bikin stres", "dosen", "laprak", "tidur", "deadline",
    "iri", "ukt berat", "kuliah tanggal segini", "nggak ada biaya",
    "gak bakat kuliah", "melepas kuliah", "menikah", "berat", "kangen sma"
]

# Labeling otomatis berdasarkan keyword
def auto_label(text):
    return int(any(kw in text.lower() for kw in keywords))

df_clean['label'] = df_clean['tweet'].apply(auto_label)

# Simpan hasil ke file baru
output_path = "/mnt/data/labeled_kuliah.csv"
df_clean.to_csv(output_path, index=False)
output_path
