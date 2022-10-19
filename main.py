# Kütüphaneler
import numpy as np
import tensorflow as tf


# Çanta içine alınması gereken ürünleri brute force bir algoritmayla çözecek olan fonksiyon
def brute_force_knapsack(agirliklar, degerler, kapasite):
    urun_sayisi = agirliklar.shape[0]
    picks_space = 2 ** urun_sayisi
    en_iyi_deger = -1
    en_iyi_secimler = np.zeros(urun_sayisi)
    for p in range(picks_space):
        secilenler = [int(c) for c in f"{p:0{urun_sayisi}b}"]
        deger = np.dot(degerler, secilenler)
        agirlik = np.dot(agirliklar, secilenler)
        if agirlik <= kapasite and deger > en_iyi_deger:
            en_iyi_deger = deger
            en_iyi_secimler = secilenler
    return en_iyi_secimler



# İçinde beş tane ürün bulunacak olan cantayı random sayılarla olusturan fonksiyon
# Sonuclar brute force bir algoritma kullanan fonksiyon sayesinde bulunacaktır
def canta_olustur():
    urun_sayisi = 5
    agirliklar = np.random.randint(1, 45, urun_sayisi)
    degerler = np.random.randint(1, 99, urun_sayisi)
    kapasite = np.random.randint(1, 99)
    cevaplar = brute_force_knapsack(agirliklar, degerler, kapasite)
    return agirliklar, degerler, kapasite, cevaplar


#seçilen urunlerin fiyatları ile optimum çözüm arasındaki ortalama farkı değerlendirir.
def metric_overprice(input_degerler):
    def overpricing(y_true, y_pred):
        y_pred = tf.keras.backend.round(y_pred)
        return tf.keras.backend.mean(tf.keras.backend.batch_dot(y_pred, input_degerler, 1) - tf.keras.backend.batch_dot(y_true, input_degerler, 1))

    return overpricing


#bu metotta seçilen öğelerin ağırlıklarının toplamı ile sırt çantasının kapasitesi arasındaki pozitif fark bulunur.
def metric_space_violation(input_agirliklar):
    def space_violation(y_true, y_pred):
        y_pred = tf.keras.backend.round(y_pred)
        return tf.keras.backend.mean(tf.keras.backend.maximum(tf.keras.backend.batch_dot(y_pred, input_agirliklar, 1) - 1, 0))

    return space_violation


def metric_pick_count():
    def pick_count(y_true, y_pred):
        y_pred = tf.keras.backend.round(y_pred)
        return tf.keras.backend.mean(tf.keras.backend.sum(y_pred, -1))

    return pick_count


# Hem verileri hem de sonda bulunması gereken sonucu verdiğimiz için supervised model kullanmış oluyoruz
# Bu modeli oluşturup geri döndürecek olan fonksiyon
# Fonksiyon içinde tensorflow kütüphanesini kullanıyoruz
def supervised_model():
    urun_sayisi = 5
    input_agirliklar = tf.keras.Input((urun_sayisi,), name="Weights")
    input_degerler= tf.keras.Input((urun_sayisi,), name="Prices")
    concat_agirlik_deger = tf.keras.layers.Concatenate(name="Concatenate")([input_agirliklar, input_degerler])
    secimler = tf.keras.layers.Dense(urun_sayisi ** 2 + urun_sayisi * 2, activation="sigmoid", name="Hidden")(concat_agirlik_deger)
    secimler = tf.keras.layers.Dense(urun_sayisi, activation="sigmoid", name="Output")(secimler)
    model = tf.keras.Model(inputs=[input_agirliklar, input_degerler], outputs=[secimler])
    model.compile("adam",
                  tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.binary_accuracy, metric_space_violation(input_agirliklar),
                           metric_overprice(input_degerler), metric_pick_count()])
    return model


# Eğitim ve test için kullacağımız verileri oluşturuyoruz
# Fonksiyon içinde canta oluşturacak fonksiyonu çağırırız,
# veri sayisi bir çantanın içinde olacak ürün sayısıdır
# Fonksiyon oranları ve cevapları geri döndürür
def create_knapsack_dataset(canta_sayisi):
    liste1 = []  # ağırlıkların kapasiteye oranını tutar
    liste2 = []  # değerlerin maximum değere olan oranını tutar
    liste3 = []  # cevapları tutar
    for _ in range(canta_sayisi):  # verilen sayı kadar canta üretilir
        agirliklar, degerler, kapasite, cevaplar = canta_olustur()
        liste1.append(agirliklar / kapasite)
        liste2.append(degerler / degerler.max())
        liste3.append(cevaplar)
    return [np.array(liste1), np.array(liste2)], np.array(liste3)


# Eğitimin gerçekleşeceği fonksiyon
# Eğitim ve test verilerini parametre olarak göndeririz
def train_knapsack(model, train_x, train_y, test_x, test_y):

    # fit fonksiyonuyla eğitimi gerçekleştiririz
    # model eğitildikçe her epok sonunda çıkan sonuçlar ekrana yazdırılır
    model.fit(train_x, train_y, epochs=256, verbose=1,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor="binary_accuracy", save_best_only=True,save_weights_only=True)])
    model.load_weights("best_model.h5")
    egitim_sonuclari = model.evaluate(train_x, train_y, 64, 0)
    test_sonuclari = model.evaluate(test_x, test_y, 64, 0)
    print("\nSupervised Model'in Sonuçları           Eğitim  Test")
    print("---------------------------------------------------")
    print(f"Loss:                        \t\t\t{egitim_sonuclari[0]:.2f} / {test_sonuclari[0]:.2f}")
    print(f"Binary accuracy:             \t\t\t{egitim_sonuclari[1]:.2f} / {test_sonuclari[1]:.2f}")
    print(f"Space violation:             \t\t\t{egitim_sonuclari[2]:.2f} / {test_sonuclari[2]:.2f}")
    print(f"Overpricing:                 \t\t\t{egitim_sonuclari[3]:.2f} / {test_sonuclari[3]:.2f}")
    print(f"Pick count:                  \t\t\t{egitim_sonuclari[4]:.2f} / {test_sonuclari[4]:.2f}")


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    train_x, train_y = create_knapsack_dataset(1000)
    test_x, test_y = create_knapsack_dataset(200)
    model = supervised_model()
    train_knapsack(model, train_x, train_y, test_x, test_y)
