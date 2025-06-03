import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer


class TextPreprocessor:
    def __init__(self):
        # Gerekli NLTK veri setlerini indir
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        # Türkçe stemmer
        self.stemmer = TurkishStemmer()

        # Türkçe stopwords listesini oluştur
        try:
            self.stop_words = set(stopwords.words('turkish'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('turkish'))

        # Ekstra stopwords ekle
        self.additional_stopwords = {
            # Genel bağlaçlar ve edatlar
            "bir", "ve", "bu", "şu", "ama", "ancak", "için", "ile", "de", "da", "ki", "den", "dan",
            "te", "ta", "mi", "mı", "mu", "mü", "ne", "ya", "ise", "ama", "fakat", "lakin", "yani",
            
            # Soru kelimeleri
            "nasıl", "neden", "niçin", "hangi", "kim", "ne", "nerede", "gibi", "kadar",
            
            # Yaygın fiiller ve zarflar
            "olmak", "etmek", "yapmak", "göre", "değil", "olarak", "üzere", "rağmen",
            "aynı", "oldu", "olacak", "yapıyor", "ediyor", "bulunuyor", "görüyor",
            
            # Zaman ifadeleri
            "şimdi", "sonra", "önce", "daha", "artık", "henüz", "bugün", "yarın", "dün",
            
            # Miktar ifadeleri
            "çok", "az", "biraz", "fazla", "tüm", "bütün", "hiç",
            
            # Yaygın ekonomi terimleri (anlam taşımayan)
            "göre", "oldu", "olacak", "arttı", "düştü", "geçti", "kaldı"
        }
        self.stop_words.update(self.additional_stopwords)

    def preprocess_text(self, text):
        """Metni temizleyip ön işler."""
        if not text or not isinstance(text, str):
            return ""

        # Küçük harfe çevirme
        text = text.lower()

        # URL'leri temizle
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Sayıları ve özel karakterleri temizleme (Türkçe karakter desteği ile)
        text = re.sub(r'[^a-zA-ZğüşıöçĞÜŞİÖÇ\s]', '', text)

        # Tokenizasyon
        tokens = word_tokenize(text)
        
        # Stopword temizleme ve stemming (geliştirilmiş mantık)
        processed_tokens = []
        for word in tokens:
            if word not in self.stop_words:
                # 2 karakterden kısa kelimeleri ele
                if len(word) <= 2:
                    continue
                    
                # Stemming uygula
                stemmed = self.stemmer.stem(word)
                # Çok kısa stem'leri ele (muhtemelen hatalı stemming)
                if len(stemmed) > 2:
                    processed_tokens.append(stemmed)

        return " ".join(processed_tokens)

    def preprocess_dataframe(self, df, column_name):
        """Bir DataFrame içindeki belirli bir kolonu işler."""
        if column_name not in df.columns:
            raise ValueError(f"'{column_name}' sütunu DataFrame içinde bulunamadı.")

        df[column_name] = df[column_name].astype(str).apply(self.preprocess_text)
        return df

    def get_text_statistics(self, df, column_name):
        """DataFrame içindeki metin sütunu hakkında istatistik çıkarır."""
        if column_name not in df.columns:
            raise ValueError(f"'{column_name}' sütunu DataFrame içinde bulunamadı.")

        stats = {
            "toplam_tweet": len(df),
            "bos_tweet": df[column_name].str.strip().eq('').sum(),
            "ortalama_kelime": df[column_name].str.split().str.len().mean()
        }
        return stats
