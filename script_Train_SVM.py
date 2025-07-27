import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt




# 1. Baca data CSV
df = pd.read_csv("Dataterakhir.csv")  # Ganti nama file sesuai

# 2. Cek dan pastikan kolom yang dipakai ada
assert 'tweet' in df.columns and 'label' in df.columns, "Pastikan ada kolom 'tweet' dan 'label'!"

# 3. Preprocessing teks (optional tapi aman)
df['tweet'] = df['tweet'].astype(str).str.lower()



# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['tweet'])
y = df['label']

# 5. Split data: train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LinearSVC(class_weight='balanced')


# 6. Latih model SVM
model = LinearSVC()
model.fit(X_train, y_train)

# 7. Evaluasi hasil
y_pred = model.predict(X_test)

# Prediksi
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()