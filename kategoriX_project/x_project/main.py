import joblib
import os
import pandas as pd
import numpy as np
from collections import Counter
from preprocessor import TextPreprocessor
from tfidf_processor import TfidfProcessor
from word2vec_processor import Word2VecProcessor
from category_terms import CategoryTerms
from model_trainer import ModelTrainer
# from tweet_classifier_test import TweetClassifierTest


from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__, template_folder='.')

def load_categorized_tweets():
    """Kategorize edilmiş tweetleri yükle"""
    try:
        print("Tweetler yükleniyor...")
        with open('categorized_tweets.json', 'r', encoding='utf-8') as f:
            tweets = json.load(f)
            print(f"Toplam {len(tweets)} tweet bulundu.")
            
            # Tweetleri kategorilere göre düzenle
            categorized = {
                'siyaset': [],
                'saglik': [],
                'spor': [],
                'teknoloji': []
            }
            
            # Kategori eşleştirme sözlüğü
            category_mapping = {
                'Yaşam': 'siyaset',
                'YAŞAM': 'siyaset',
                'Siyaset': 'siyaset',
                'SİYASET': 'siyaset',
                'Sağlık': 'saglik',
                'sağlık': 'saglik',
                'SAGLIK': 'saglik',
                'SAĞLIK': 'saglik',
                'Spor': 'spor',
                'SPOR': 'spor',
                'Teknoloji': 'teknoloji',
                'TEKNOLOJİ': 'teknoloji',
                'TEKNOLOJI': 'teknoloji'
            }
            
            print("Mevcut kategoriler:", set(tweet['category'] for tweet in tweets))
            
            for tweet in tweets:
                category = tweet['category']
                normalized_category = category_mapping.get(category, '').lower()
                print(f"Kategori dönüşümü: {category} -> {normalized_category}")
                
                if normalized_category in categorized:
                    tweet_data = {
                        'username': tweet.get('username', 'Anonim'),
                        'content': tweet['text'],
                        'date': tweet.get('date', ''),
                        'like_count': tweet.get('like_count', 0)
                    }
                    categorized[normalized_category].append(tweet_data)
                else:
                    print(f"Uyarı: '{category}' kategorisi eşleştirilemedi")
            
            # Her kategorideki tweet sayısını göster
            for cat, tweets in categorized.items():
                print(f"{cat}: {len(tweets)} tweet")
                
            return categorized
    except Exception as e:
        print(f"Tweet yükleme hatası: {e}")
        return None

def main():
    # Ana sayfayı göster
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Tweet verilerini API endpoint'i olarak sun
    @app.route('/api/tweets')
    def get_tweets():
        tweets = load_categorized_tweets()
        if tweets:
            return jsonify(tweets)
        return jsonify({'error': 'Tweetler yüklenemedi'})
    
    # Web sunucusunu başlat
    print("Web sunucusu başlatılıyor...")
    print("Tarayıcınızda http://localhost:5000 adresini açın")
    app.run(debug=True, port=5000)


def model_egitimi():
    # Veri ön işleme sınıflarını başlat
    preprocessor = TextPreprocessor()
    word2vec_processor = Word2VecProcessor()
    tfidf_processor = TfidfProcessor(
        min_df=2,
        max_df=0.6,  # Daha fazla kelime için arttırıldı
        max_features=1000,  # Daha fazla özellik
        ngram_range=(1, 3)  # 1-3 gram kullanımı
    )
    model_trainer = ModelTrainer()

    # Dosyanın var olup olmadığını kontrol et
    file_path = "temizlenmis_veri.xlsx"
    if not os.path.exists(file_path):
        print(f"Hata: '{file_path}' dosyası bulunamadı!")
        return

    try:
        # Excel dosyasını oku
        df = pd.read_excel(file_path)
        print("Veri seti başarıyla yüklendi. Toplam örnek sayısı:", len(df))

        # Gerekli sütunların var olup olmadığını kontrol et
        required_columns = {'tweets', 'etiket'}
        if not required_columns.issubset(df.columns):
            print(f"Hata: Veri setinde şu sütunlar eksik: {required_columns - set(df.columns)}")
            return

        # Veriyi ön işle
        print("\nMetinler ön işleniyor...")
        df = preprocessor.preprocess_dataframe(df, 'tweets')
        print("Ön işleme tamamlandı.")

        # Word2Vec modelini eğit
        print("\nWord2Vec modeli eğitiliyor...")
        sentences = [tweet.split() for tweet in df['tweets']]
        word2vec_model = word2vec_processor.train_model(sentences)
        print("Word2Vec modeli eğitimi tamamlandı.")

        # TF-IDF matrisini oluştur ve CategoryTerms ağırlıklarını uygula
        print("\nTF-IDF matrisi oluşturuluyor ve kategori terimleri ağırlıkları uygulanıyor...")
        X = tfidf_processor.fit_transform(df['tweets'].tolist())

        # Word2Vec semantic scores'u hesapla
        print("\nWord2Vec semantic scores hesaplanıyor...")
        word_freq = Counter(' '.join(df['tweets']).split())
        semantic_scores = word2vec_processor.calculate_semantic_importance_scores(
            feature_names=tfidf_processor.vectorizer.get_feature_names_out(),
            word_freq=word_freq,
            total_docs=len(df)
        )

        # TF-IDF matrisini semantic scores ile güncelle
        feature_names = tfidf_processor.vectorizer.get_feature_names_out()
        for idx, term in enumerate(feature_names):
            if term in semantic_scores:
                X[:, idx] *= semantic_scores[term]

        print("TF-IDF matrisi oluşturuldu ve ağırlıklar uygulandı. Boyut:", X.shape)

        # Hedef değişkeni ayır
        y = df['etiket']

        # Modeli eğit ve değerlendir
        print("\nModel eğitiliyor...")
        X_train, X_test, y_train, y_test = model_trainer.split_data(X, y)
        model = model_trainer.train_model(X_train, y_train)
        model_trainer.evaluate_model(X_test, y_test)
        
        # Sonuçları göster
        model_trainer.print_metrics()

        # Modeli kaydet
        print("\nModeller kaydediliyor...")
        joblib.dump(model, 'tweet_model.joblib')
        joblib.dump(tfidf_processor.vectorizer, 'tfidf_vectorizer.joblib')
        joblib.dump(model_trainer.label_encoder, 'label_encoder.joblib')
        word2vec_model.save("word2vec_model.model")
        print("Modeller başarıyla kaydedildi.")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

def yeni_veri_ekle():
    """Yeni verileri işleyip temizlenmis_veri.xlsx dosyasına ekler."""
    try:
        # Kullanıcıdan yeni veri dosyasının adını al
        dosya_adi = input("Yeni veri dosyasının adını giriniz (örn: yeni_veriler.xlsx): ")
        
        if not os.path.exists(dosya_adi):
            print(f"Hata: '{dosya_adi}' dosyası bulunamadı!")
            return

        # Yeni verileri oku
        yeni_df = pd.read_excel(dosya_adi)
        
        # Gerekli sütunların kontrolü
        required_columns = {'tweets', 'etiket'}
        if not required_columns.issubset(yeni_df.columns):
            print(f"Hata: Veri setinde şu sütunlar eksik: {required_columns - set(yeni_df.columns)}")
            return

        # Veri ön işleme
        preprocessor = TextPreprocessor()
        print("\nYeni veriler ön işleniyor...")
        yeni_df = preprocessor.preprocess_dataframe(yeni_df, 'tweets')
        print("Ön işleme tamamlandı.")

        # Mevcut temizlenmiş veriyi oku (eğer varsa)
        temiz_veri_path = "temizlenmis_veri_new.xlsx"
        if os.path.exists(temiz_veri_path):
            mevcut_df = pd.read_excel(temiz_veri_path)
            # Yeni verileri mevcut verilerle birleştir
            birlesik_df = pd.concat([mevcut_df, yeni_df], ignore_index=True)
        else:
            birlesik_df = yeni_df

        # Birleştirilmiş veriyi kaydet
        birlesik_df.to_excel(temiz_veri_path, index=False)
        print(f"\nYeni veriler başarıyla '{temiz_veri_path}' dosyasına eklendi.")
        print(f"Toplam veri sayısı: {len(birlesik_df)}")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
def test_arayuzu():
    """Test arayüzünü başlatır ve kullanıcının tweet'leri test etmesini sağlar."""
    try:
        # Test sınıfını başlat
        classifier = TweetClassifierTest()

        print("\nNot: Ana menüye dönmek için 'exit' yazın.")

        while True:
            # Kullanıcıdan tweet al
            tweet = input("\nLütfen test etmek için bir tweet giriniz: ")

            # Çıkış kontrolü
            if tweet.lower() == 'exit':
                break

            try:
                # Yeni predict metodunu kullan
                category = classifier.predict(tweet)
                print(f"\n→ Tahmini Kategori: {category}\n")
            except Exception as e:
                print(f"Tahmin hatası: {e}\n")

    except Exception as e:
        print(f"Test arayüzü başlatma hatası: {e}")


if __name__ == "__main__":
    main()