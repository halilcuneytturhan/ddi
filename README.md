# HouseMD-Intent: Medikal Diyalog Niyeti Sınıflandırma Sistemi

Bu proje, **House MD veri seti** kullanılarak geliştirilen bir doğal dil işleme uygulamasıdır. Amaç, kullanıcı tarafından girilen medikal içerikli bir diyaloğun hangi konuşma niyetine ait olduğunu tahmin etmektir.

Proje kapsamında veri temizleme, özellik çıkarımı, sınıflandırma modeli eğitimi ve Streamlit tabanlı web arayüzü geliştirilmiştir.

---

## Proje Amacı

Bu çalışmanın amacı, House MD dizisinden oluşturulan medikal diyalog veri seti üzerinde bir **konuşma niyeti sınıflandırma modeli** geliştirmektir.

Model, girilen metni aşağıdaki intent sınıflarından birine atar:

- açıklama
- hipotez
- soru
- talimat
- tanı
- tedavi
- değerlendirme
- test/prosedür

Örnek:

```text
Girdi:
Hemen MR çekin ve kan testlerini kontrol edin.

Model çıktısı:
talimat
```

---

## Kullanılan Teknolojiler

Projede kullanılan temel teknolojiler:

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Calibrated SVM
- Joblib
- Streamlit

---

## Veri Seti

Projede kullanılan veri seti, House MD dizisindeki medikal diyaloglardan oluşturulmuştur.

Veri setinde birçok sütun bulunmasına rağmen bu proje kapsamında temel olarak şu iki sütun kullanılmıştır:

| Sütun | Açıklama |
|---|---|
| `text` | Diyalog/metin verisi |
| `Intent` | Metnin konuşma niyeti etiketi |

Model eğitimi için `Intent` sütunu hedef değişken olarak kullanılmıştır.

---

## Veri Ön İşleme

Veri ön işleme aşamasında aşağıdaki işlemler yapılmıştır:

- Eksik `text` ve `Intent` değerleri temizlenmiştir.
- Metinler küçük harfe çevrilmiştir.
- Gereksiz boşluklar temizlenmiştir.
- Gereksiz özel karakterler kaldırılmıştır.
- Intent etiketleri standart hale getirilmiştir.
- Aynı anlama gelen etiketler tek sınıf altında toplanmıştır.
- Çok dengesiz sınıflar için yumuşak sınıf dengeleme uygulanmıştır.

Örnek etiket düzenleme:

```text
Açıklama → açıklama
aciklama → açıklama
teşhis → tanı
teshis → tanı
prosedur → test/prosedür
```

---

## Modelleme

Metin verileri doğrudan makine öğrenmesi modellerine verilemediği için önce **TF-IDF** yöntemiyle sayısal vektörlere dönüştürülmüştür.

Daha sonra sınıflandırma modeli olarak **Calibrated SVM** kullanılmıştır.

Genel model akışı:

```text
Metin
↓
Metin temizleme
↓
TF-IDF vektörleştirme
↓
Calibrated SVM modeli
↓
Intent tahmini
```

---

## Model Performansı

Model, eğitim ve test verisi olarak ayrılan veri üzerinde değerlendirilmiştir.

Kullanılan değerlendirme metrikleri:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Modelde bazı sınıfların başarı oranı daha yüksek, bazı sınıfların başarı oranı daha düşük çıkmıştır. Bunun temel nedenleri:

- Bazı intent sınıflarının veri sayısının az olması
- Bazı sınıfların dilsel olarak birbirine benzemesi
- Kısa cümlelerde bağlam bilgisinin sınırlı olması

Özellikle `soru`, `talimat`, `hipotez` ve `tedavi` sınıflarında daha başarılı sonuçlar alınmıştır.

---

## Web Arayüzü

Proje, **Streamlit** ile web arayüzüne dönüştürülmüştür.

Kullanıcı arayüzünde:

- Kullanıcı bir medikal diyalog/metin girer.
- Sistem metni temizler.
- TF-IDF ile sayısal forma dönüştürür.
- Eğitilmiş model ile intent tahmini yapar.
- Tahmin edilen intent ve güven skoru ekranda gösterilir.

---

## Etik Not

Bu proje yalnızca eğitim ve akademik çalışma amacıyla geliştirilmiştir. Doğal Dil işleme dersinde yapacağımız projenin ana kaynak veri seti ders içerisinde kendimiz oluşturduğumuz veri setleridir. 

Modelin verdiği çıktılar gerçek tıbbi karar, tanı veya tedavi önerisi olarak kullanılmamalıdır. Veri seti House MD dizisinden oluşturulduğu için kurgu içeriklere dayanmaktadır. Bu nedenle model sonuçları gerçek doktorun söylediği şeyler için uygun değildir.

---

## Geliştirilebilecek Yönler

İlerleyen aşamalarda proje şu şekilde geliştirilebilir:

- Daha büyük ve dengeli veri seti kullanılabilir.
- BERTurk veya multilingual BERT gibi transformer tabanlı modeller denenebilir.
- Daha ayrıntılı hata analizi yapılabilir.
- Kullanıcı arayüzü geliştirilebilir.
- Modelin en yakın tahminleri de gösterilebilir.
- Duygu analizi veya alaycılık tespiti gibi ek NLP görevleri eklenebilir.

---

## Proje Bilgileri

Bu proje, Doğal Dil İşleme dersi kapsamında hazırlanmıştır.

**Proje:** HouseMD-Intent  
**Konu:** Medikal diyaloglarda konuşma niyeti sınıflandırması  
**Model:** TF-IDF + Calibrated SVM  
**Arayüz:** Streamlit
