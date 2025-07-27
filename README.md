# 📌 Deteksi Stres Akademik Mahasiswa dari Tweet Menggunakan TF-IDF + SVM  

Proyek ini bertujuan untuk **mengidentifikasi tweet yang mengindikasikan stres akademik mahasiswa** dengan pendekatan **machine learning** menggunakan **TF-IDF Vectorizer** dan **Linear Support Vector Machine (SVM)**.  

---

## 📂 Fitur Utama  
✅ Membersihkan data tweet (hapus link, mention, hashtag, newline)  
✅ Menghapus duplikasi tweet  
✅ Memberi label otomatis berdasarkan daftar kata kunci stres akademik  
✅ Melatih model **SVM** menggunakan representasi **TF-IDF**  
✅ Menampilkan **Confusion Matrix** dan **classification report**  

---

## 📁 Struktur Proyek  
📦 project-folder
├── Kuliahpart2.csv # Dataset mentah (tweet mentah)
├── Dataterakhir.csv # Dataset hasil labeling
├── labeled_kuliahpart2.csv # Output hasil labeling otomatis
├── script_clean_label.py # Script untuk membersihkan & memberi label tweet
├── script_train_svm.py # Script untuk training model SVM
└── README.md # Dokumentasi proyek

---

## ⚙️ 1. Persiapan  
### 📌 Instalasi Library  
Jalankan perintah berikut di terminal:  
```bash
pip install pandas scikit-learn seaborn matplotlib
