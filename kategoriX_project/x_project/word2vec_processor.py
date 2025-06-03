import numpy as np
from gensim.models import Word2Vec
import math


class Word2VecProcessor:
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def train_model(self, sentences):
        """
        Word2Vec modelini eğitir ve kaydeder.
        """
        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1,  # Skip-gram modeli
            hs=0,  # Negative sampling yöntemi
            negative=5,
            alpha=0.025,
            min_alpha=0.0001
        )

        self.model.build_vocab(sentences, progress_per=1000)
        self.model.train(sentences, total_examples=len(sentences), epochs=10)

        self.model.save("word2vec_model.model")
        return self.model

    def load_model(self, model_path="word2vec_model.model"):
        try:
            self.model = Word2Vec.load(model_path)
            return self.model
        except FileNotFoundError:
            print("Hata: Model dosyası bulunamadı. Lütfen önce modeli eğitin.")
            return None

    def calculate_semantic_importance_scores(self, feature_names, word_freq, total_docs):

        if self.model is None:
            raise ValueError("Word2Vec modeli yüklenmemiş. Önce train_model() veya load_model() çağırılmalıdır.")

        semantic_scores = {}
        vocabulary = set(self.model.wv.index_to_key)

        for term in feature_names:
            if term not in vocabulary:
                semantic_scores[term] = 1.0  # Varsayılan değer
                continue

            # Terim vektörünü al
            term_vector = self.model.wv[term]

            # Terimin doküman frekansını hesapla (word_freq'den)
            doc_freq = word_freq.get(term, 1)  # En az 1 olsun
            
            # IDF benzeri bir skor hesapla
            idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1

            # Terimin diğer terimlerle olan benzerliklerini hesapla
            similarity_sum = 0
            similarity_count = 0

            # En yakın 10 kelimeyi bul
            try:
                similar_words = self.model.wv.most_similar(term, topn=10)
                for similar_word, similarity in similar_words:
                    if similar_word in word_freq:
                        similarity_sum += similarity
                        similarity_count += 1
            except KeyError:
                pass  # Kelime modelde yoksa atla

            # Ortalama benzerlik skorunu hesapla
            avg_similarity = (similarity_sum / similarity_count) if similarity_count > 0 else 0.5

            # Son semantik skoru hesapla
            semantic_score = (0.4 * idf + 0.6 * avg_similarity)
            semantic_scores[term] = semantic_score

        return semantic_scores
