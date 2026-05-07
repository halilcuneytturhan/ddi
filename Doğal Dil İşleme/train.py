import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


# =========================
# 1. Temizlenmiş dataseti oku
# =========================

data_path = "data/cleaned_intent_dataset.csv"
df = pd.read_csv(data_path)


# =========================
# Yumuşak sınıf dengeleme
# =========================





print("Dataset boyutu:", df.shape)
print("\nSınıf dağılımı:")
print(df["label"].value_counts())

# =========================
# Yumuşak sınıf dengeleme
# =========================

max_per_class = 500

df = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=42), include_groups = False)
)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nYumuşak dengelenmiş dataset boyutu:", df.shape)
print("\nYumuşak dengelenmiş sınıf dağılımı:")
print(df["label"].value_counts())


# =========================
# 2. Giriş ve hedef değişkenleri ayır
# =========================

X = df["text"]
y = df["label"]


# =========================
# 3. Train-test ayrımı
# =========================
# train: modelin öğreneceği veri
# test: modelin daha önce görmediği veri

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nEğitim veri sayısı:", X_train.shape[0])
print("Test veri sayısı:", X_test.shape[0])


# =========================
# 4. TF-IDF ile metinleri sayısallaştır
# =========================

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.90,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF eğitim matris boyutu:", X_train_tfidf.shape)
print("TF-IDF test matris boyutu:", X_test_tfidf.shape)


# =========================
# 5. Modeli oluştur ve eğit
# =========================


base_model = LinearSVC(
    class_weight="balanced",
    C=1.0,
    random_state=42
)

model = CalibratedClassifierCV(
    estimator=base_model,
    cv=5
)

model.fit(X_train_tfidf, y_train)

print("\nModel eğitimi tamamlandı.")


# =========================
# 6. Test verisi üzerinde tahmin yap
# =========================

y_pred = model.predict(X_test_tfidf)


# =========================
# 7. Model performansını ölç
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# 8. Model ve vectorizer dosyalarını kaydet
# =========================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/intent_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel dosyaları kaydedildi:")
print("models/intent_model.pkl")
print("models/tfidf_vectorizer.pkl")


# =========================
# 9. Basit test tahminleri
# =========================

sample_texts = [
    "Hastanın ateşi yükseliyor ve nefes almakta zorlanıyor.",
    "Bu lupus olabilir mi?",
    "Hemen MR çekin.",
    "Tedaviye antibiyotikle başlayalım.",
    "Kan testinin sonucunu değerlendirmemiz gerekiyor."
]

sample_vectors = vectorizer.transform(sample_texts)
sample_predictions = model.predict(sample_vectors)

print("\nÖrnek tahminler:")
for text, pred in zip(sample_texts, sample_predictions):
    print(f"Metin: {text}")
    print(f"Tahmin edilen intent: {pred}")
    print("-" * 50)