import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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
      .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=42))
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
    min_df=2,
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


candidate_models = {
    "Calibrated SVM": CalibratedClassifierCV(
        estimator=LinearSVC(
            class_weight="balanced",
            C=1.0,
            random_state=42
        ),
        cv=5
    ),
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        C=2.0,
        max_iter=2000
    )
}

comparison_results = []
best_model_name = None
best_model = None
best_score = -1
best_accuracy = -1
best_predictions = None

print("\nModel karsilastirmasi basladi.")

for model_name, candidate_model in candidate_models.items():
    print(f"\nEgitilen model: {model_name}")
    candidate_model.fit(X_train_tfidf, y_train)

    candidate_predictions = candidate_model.predict(X_test_tfidf)
    candidate_accuracy = accuracy_score(y_test, candidate_predictions)
    candidate_macro_f1 = f1_score(y_test, candidate_predictions, average="macro")

    comparison_results.append({
        "model": model_name,
        "accuracy": candidate_accuracy,
        "macro_f1": candidate_macro_f1
    })

    print(f"Accuracy: {candidate_accuracy:.4f}")
    print(f"Macro F1: {candidate_macro_f1:.4f}")

    if (
        candidate_macro_f1 > best_score
        or (candidate_macro_f1 == best_score and candidate_accuracy > best_accuracy)
    ):
        best_model_name = model_name
        best_model = candidate_model
        best_score = candidate_macro_f1
        best_accuracy = candidate_accuracy
        best_predictions = candidate_predictions

model = best_model
y_pred = best_predictions
setattr(model, "selected_model_name_", best_model_name)

print("\nModel karsilastirma ozeti:")
for result in comparison_results:
    print(
        f"{result['model']}: "
        f"accuracy={result['accuracy']:.4f}, "
        f"macro_f1={result['macro_f1']:.4f}"
    )

print(f"\nSecilen model: {best_model_name}")

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
