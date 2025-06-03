from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from category_terms import CategoryTerms


class TfidfProcessor:
    def __init__(self, min_df=5, max_df=0.7, max_features=1000, ngram_range=(1, 2)):
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.feature_names = None
        self.category_terms = CategoryTerms()

    def fit_transform(self, texts):
        """TF-IDF dönüşümünü gerçekleştirir ve döndürür."""
        if not texts or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Metin listesi geçerli değil. Boş olmamalı ve tüm elemanları string olmalıdır.")

        # CategoryTerms'den kelime ve ağırlıkları al
        terms_dict = self.category_terms.get_terms()
        vocabulary = {term: idx for idx, term in enumerate(terms_dict.keys())}

        self.vectorizer = TfidfVectorizer(
            vocabulary=vocabulary,  # Önceden belirlenmiş kelime dağarcığını kullan
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            norm='l2',
            lowercase=False  # Metinler zaten preprocessor tarafından küçük harfe çevrildiği için False
        )

        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # CategoryTerms'den gelen ağırlıkları uygula
        term_weights = np.ones(len(self.feature_names))
        for idx, term in enumerate(self.feature_names):
            if term in terms_dict:
                term_weights[idx] = terms_dict[term]

        # Ağırlıkları TF-IDF matrisine uygula
        X_array = X.toarray()
        X_weighted = np.multiply(X_array, term_weights)

        return X_weighted

    def apply_semantic_weights(self, X, semantic_importance, alpha=0.7):
        """TF-IDF skorlarına semantik ağırlıkları uygular."""
        if self.feature_names is None:
            raise ValueError("Özellik isimleri tanımlanmadı. Önce fit_transform() çağırılmalıdır.")

        semantic_weights = np.ones(len(self.feature_names))
        for i, term in enumerate(self.feature_names):
            if term in semantic_importance:
                semantic_weights[i] += alpha * semantic_importance[term]

        return np.multiply(X, semantic_weights)

    def get_feature_names(self):
        """Özellik isimlerini döndürür."""
        if self.feature_names is None:
            raise ValueError("Özellik isimleri tanımlanmadı. Önce fit_transform() çağırılmalıdır.")
        return self.feature_names

    def transform(self, texts):
        """Yeni metinleri dönüştürür."""
        if self.vectorizer is None:
            raise ValueError("Vektörleştirici henüz eğitilmedi. Önce fit_transform() çağırılmalıdır.")
        
        X = self.vectorizer.transform(texts)
        
        # CategoryTerms ağırlıklarını uygula
        terms_dict = self.category_terms.get_terms()
        term_weights = np.ones(len(self.feature_names))

        for idx, term in enumerate(self.feature_names):
            if term in terms_dict:
                term_weights[idx] = terms_dict[term]
                
        X_array = X.toarray()
        X_weighted = np.multiply(X_array, term_weights)
        
        return X_weighted
