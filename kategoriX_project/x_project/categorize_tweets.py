import json
import joblib
from preprocessor import TextPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer

def load_tweets():
    """tweets.json dosyasından tweetleri yükle"""
    try:
        with open('tweets.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading tweets: {e}")
        return {}

def save_predictions(predictions):
    """Tahminleri yeni bir JSON dosyasına kaydet"""
    # Sadece tweet metni ve tahmin edilen kategoriyi içeren basitleştirilmiş bir yapı
    simplified_predictions = []
    
    for username, tweets in predictions.items():
        for tweet in tweets:
            simplified_predictions.append({
                'text': tweet['text'],
                'category': tweet['predicted_category']
            })
    
    with open('categorized_tweets.json', 'w', encoding='utf-8') as f:
        json.dump(simplified_predictions, f, ensure_ascii=False, indent=4)
        print(f"\nSonuçlar 'categorized_tweets.json' dosyasına kaydedildi.")

def main():
    # Model ve vectorizer'ı yükle
    try:
        print("Model ve vectorizer yükleniyor...")
        model = joblib.load('trained_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        
        # Vectorizer'ın özellik setini kontrol et
        if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
            print("Hata: Vectorizer'ın özellik seti bulunamadı!")
            return
            
        print("Model ve vectorizer başarıyla yüklendi.")
        print(f"Vectorizer özellik sayısı: {len(vectorizer.vocabulary_)}")
        print(f"Model beklenen özellik sayısı: {model.coef_.shape[1]}")
        
    except FileNotFoundError as e:
        print("Hata: Model veya vectorizer dosyası bulunamadı!")
        print("Lütfen 'trained_model.joblib' ve 'tfidf_vectorizer.joblib' dosyalarının")
        print("proje klasöründe olduğundan emin olun.")
        return
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")
        return

    # Text preprocessor'ı başlat
    preprocessor = TextPreprocessor()
    
    # Tweetleri yükle
    tweets_data = load_tweets()
    
    # Tüm tweetleri ve işlenmiş metinleri topla
    all_processed_texts = []
    tweet_mapping = []  # İşlenmiş tweet ile orijinal tweet arasındaki mapping
    
    print("Tweetler işleniyor...")
    for username, tweets in tweets_data.items():
        for tweet in tweets:
            processed_text = preprocessor.preprocess_text(tweet['text'])
            if processed_text.strip():  # Boş olmayan işlenmiş metinleri al
                all_processed_texts.append(processed_text)
                tweet_mapping.append((username, tweet))
    
    if not all_processed_texts:
        print("İşlenecek tweet bulunamadı!")
        return
    
    print(f"Toplam {len(all_processed_texts)} tweet işlendi.")
    
    # Tüm metinleri vektörize et
    print("Tweetler vektörize ediliyor...")
    try:
        # Vectorizer'ı kullanarak metinleri vektörize et ve model ile aynı özellikleri kullan
        vectorized_texts = vectorizer.transform(all_processed_texts)
        
        # Eğer özellik sayısı farklıysa, sadece model'in bildiği özellikleri seç
        if vectorized_texts.shape[1] > model.coef_.shape[1]:
            # Modelin bildiği özelliklerin indekslerini al
            feature_names = vectorizer.get_feature_names_out()
            selected_features = feature_names[:model.coef_.shape[1]]
            
            # Yeni bir vectorizer oluştur ve sadece seçili özellikleri kullan
            new_vectorizer = TfidfVectorizer(vocabulary=dict(zip(selected_features, range(len(selected_features)))))
            vectorized_texts = new_vectorizer.fit_transform(all_processed_texts)
        
        # Özellik sayısını kontrol et
        if vectorized_texts.shape[1] != model.coef_.shape[1]:
            print(f"Hata: Vektörize edilen verinin özellik sayısı ({vectorized_texts.shape[1]}) ")
            print(f"model'in beklediği özellik sayısından ({model.coef_.shape[1]}) farklı!")
            return
    except Exception as e:
        print(f"Vektörizasyon sırasında hata: {e}")
        return
    
    # Toplu tahmin yap
    print("Kategoriler tahmin ediliyor...")
    predictions = {}
    all_predictions = model.predict(vectorized_texts)
    
    # Tahminleri organize et
    for idx, (username, tweet) in enumerate(tweet_mapping):
        if username not in predictions:
            predictions[username] = []
        
        predictions[username].append({
            'text': tweet['text'],
            'processed_text': all_processed_texts[idx],
            'predicted_category': all_predictions[idx],
            'date': tweet['date']
        })
    
    # Sonuçları kaydet
    save_predictions(predictions)
    print("Tweet kategorileri başarıyla tahmin edildi ve kaydedildi!")

if __name__ == "__main__":
    main()
