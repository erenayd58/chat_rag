# Chunk Yöntemleri ve Sorgu Ekranı — Basit Anlatım

Bu doküman, sistemdeki **4 farklı chunk (parçalama) yönteminin** çalışma prensibini ve
**sorgu ekranının** bir soruya nasıl cevap ürettiğini basitçe anlatır.

---

## Chunk nedir, neden parçalıyoruz?

Bir PDF yüklendiğinde model dokümanın tamamını tek seferde okuyamaz. Bu yüzden doküman,
**chunk** adı verilen küçük, anlamlı parçalara bölünür. Soru sorulduğunda sistem tüm
dokümanı değil, soruyla en ilgili chunk'ları bulup modele verir.

Parçalamanın kalitesi her şeyi belirler: Bir tablo ortadan ikiye bölünürse ya da bir
bölümün başlığı bir parçada, içeriği başka parçada kalırsa, arama o bilgiyi bulamaz.
İşte bu yüzden 4 farklı yöntem var — her biri "nereden keselim?" sorusuna farklı cevap verir.

> **Ortak kurallar:** Doküman **bir kez** parse edilir; dört yöntem de aynı kanonik
> birimler üzerinde çalışır. Token bütçesi de hepsinde aynıdır (hedef ~700 token,
> üst sınır ~1126). Böylece yöntemler karşılaştırılırken tek fark, **sınırların nereye
> düştüğü** olur.

---

## 1) Markdown — Sabit Boyutlu Kesim (Taban Çizgisi)

**Prensip:** Metni cetvelle keser gibi böler.

- Dokümanın başlık/bölüm yapısına **bakmaz**.
- Metni sabit boyutta (~700 token) parçalara ayırır, parçalar arasında bir miktar
  **örtüşme** (~140 token) bırakır ki sınırdaki cümleler tamamen kaybolmasın.
- En hızlı ve en ucuz yöntemdir; model ya da embedding kullanmaz.

**Ne işe yarar?** Karşılaştırmalarda **taban çizgisidir**: Diğer yöntemlerin "yapıya
bakmak gerçekten fayda sağlıyor mu?" sorusunu ölçmek için vardır.

**Zayıf yanı:** Kesim yeri anlamı umursamaz — bir tabloyu ya da paragrafı ortadan
bölebilir.

---

## 2) Standard — Yapı Öncelikli Kesim

**Prensip:** Dokümanın kendi iskeletini takip eder.

- Başlıkları, bölümleri, tabloları ve listeleri tanır; **her bölüm kendi başlığı
  altında kalır**.
- Sınırlar yapının söylediği yere konur; token limitleri sadece kısıtlayıcıdır.
- Bütçeyi aşan çok büyük bölümler, **doğal dikiş yerlerinden** bölünür:
  tablo satırı, liste maddesi ya da cümle sonu — asla kelime ortasından değil.
- Embedding ya da model kullanmaz; parser hızında çalışır, maliyeti sıfırdır.

**Ne işe yarar?** Varsayılan yöntemdir. Hız/kalite dengesi en iyi olandır ve yeni
yüklenen dokümanlar aksi söylenmedikçe bununla parçalanır.

---

## 3) Deep Analysis — Modelle Doğrulanan Kesim

**Prensip:** Standard'ın işini bitirdiği yerden devam eder ve **kötü sınırları arayıp düzeltir**.

Akışı şöyledir:

1. Önce Standard ile aynı yapı-öncelikli parçalama yapılır.
2. Sonra şüpheli sınırlar (ör. bir konunun ortasından geçen kesimler) tespit edilir.
3. Kararsız kalınan yerlerde **dil modeline danışılır**: model alternatif kesim önerir.
4. Her öneri **iki kez doğrulanır** (çift sıralı doğrulayıcı) — model "evet, bu kesim
   daha iyi" diyemezse öneri reddedilir ve Standard'ın kesimi korunur.

Model yapılandırılmamışsa (API anahtarı yoksa) Deep Analysis sessizce
Standard'a dönmek yerine **deterministik kalite sözleşmesini** çalıştırır ve sonucu açıkça
"model olmadan üretildi" diye işaretler.

**Ne işe yarar?** Premium ingest modudur — en yüksek kaliteli sınırları üretir, ama
model çağrısı yaptığı için en yavaş ve en maliyetli yöntemdir.

---

## 4) Hybrid — Yapı + Anlam Benzerliği

**Prensip:** Yapıyı takip eder, ama bölmek zorunda kaldığında **anlama bakar**.

- Standard gibi başlık/bölüm yapısını izler.
- Fark şurada: Bütçeyi aşan büyük bir bölümü bölmek gerektiğinde, aday kesim
  noktalarını bir **cümle embedding modeliyle** (`intfloat/multilingual-e5-base`)
  puanlar ve **anlamın en çok değiştiği yeri** seçer.
- Yani "bu bölümü bir yerden kesmem lazım — konunun döndüğü noktadan keseyim" der.

**Ne işe yarar?** Uzun, tek başlık altında birden fazla konu barındıran bölümlerde
Standard'dan daha isabetli sınırlar üretir. LLM çağrısı yapmaz ama embedding modeli
gerektirir; model makinede indirilmemişse bu seçenek ekranda **nedeniyle birlikte**
"kullanılamaz" olarak görünür.

---

## Dört Yöntem Bir Bakışta

| Yöntem | Yapıya bakar mı? | Anlama bakar mı? | Model maliyeti | Hız |
|---|---|---|---|---|
| **Markdown** | Hayır | Hayır | Yok | En hızlı |
| **Standard** | Evet | Hayır | Yok | Hızlı |
| **Deep Analysis** | Evet | Evet (LLM doğrulamalı) | LLM çağrısı | En yavaş |
| **Hybrid** | Evet | Evet (embedding) | Embedding | Orta |

---

## Sorgu Ekranı Nasıl Çalışır?

Sorgu ekranında kullanıcı bir **bilgi tabanı** seçer, sorusunu yazar ve cevabı
**kaynaklarıyla birlikte** alır. Perde arkasında şu zincir çalışır:

```
Soru → Netleştirme → Arama (vektör + BM25) → Yeniden Sıralama → Cevap + Kaynaklar
```

### Adım adım

1. **Konuşma bağlamı** — Sohbetin son birkaç turu hatırlanır. "Peki bu oran geçen yıl
   neydi?" gibi bir soru, önceki mesajlara bakılarak neyi kastettiği anlaşılacak şekilde
   ele alınır.

2. **Sorgu netleştirme ve genişletme** — Soru arama için iyileştirilir: belirsiz
   ifadeler bağlamla netleştirilir, anahtar kelimeler çıkarılır ve gerekirse sorunun
   birkaç farklı varyasyonu üretilir (query expansion). Böylece "personel sayısı" için
   "çalışan sayısı" geçen bir bölüm de bulunabilir.

3. **Arama (retrieval)** — İlgili chunk'lar iki farklı gözle aranır:
   - **Vektör arama:** Soru embedding'e çevrilir, **anlamca** en yakın chunk'lar bulunur.
   - **BM25 (anahtar kelime):** Sorudaki kelimelerin **birebir geçtiği** chunk'lar bulunur.
   - **Hybrid:** İkisinin sonuçları birleştirilir — anlam benzerliği ve kelime eşleşmesi
     birlikte puanlanır. Varsayılan mod budur; çünkü "kredi riski" gibi teknik terimlerde
     kelime eşleşmesi, serbest sorularda ise anlam benzerliği daha güçlüdür.

4. **Yeniden sıralama (reranking)** — Bulunan adaylar bir **cross-encoder** modelinden
   geçirilir: model, soru ile her chunk'ı yan yana okuyup "bu parça bu soruyu gerçekten
   cevaplıyor mu?" diye puanlar ve en iyileri öne alır.

5. **Cevap üretimi** — En iyi chunk'lar bir bağlam paketi halinde dil modeline verilir.
   Model cevabı **yalnızca bu parçalara dayanarak** yazar.

6. **Kaynak gösterimi** — Cevabın altında dayandığı doküman bölümleri listelenir:
   hangi doküman, hangi bölüm, hangi sayfalar ve benzerlik puanı. Bir kaynağa
   tıklandığında ilgili chunk'ın tam içeriği açılır — böylece cevabın **nereden
   geldiği her zaman denetlenebilir**.

### Önemli bir prensip

Chunk'lama **yalnızca yükleme sırasında** yapılır. Soru sorulduğunda hiçbir parçalama,
öneri ya da doğrulama çalışmaz — sorgu ekranı, yükleme anında indekslenmiş hazır
chunk'ları okur. Bu sayede sorgular hızlıdır ve aynı soruya her zaman aynı indeks
üzerinden cevap aranır.
