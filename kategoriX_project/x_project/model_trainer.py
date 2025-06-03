from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


class ModelTrainer:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.metrics = {}
        self.classification_report = None
        self.confusion_matrix = None

    def split_data(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded  # Sınıf dengesini korumak için
        )
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Lojistik regresyon modelini eğitir"""
        self.model = LogisticRegression(
            max_iter=1000,  # Yüksek iterasyon sayısı
        )
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        """Yeni veriler için tahmin yapar"""
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş!")
        return self.model.predict(X)

    def evaluate_model(self, X_test, y_test):
        """Model performansını değerlendirir"""
        y_pred = self.model.predict(X_test)

        # Metrikleri hesapla
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        # Detaylı sınıflandırma raporu
        self.classification_report = classification_report(y_test, y_pred)

        # Karmaşıklık matrisi
        self.confusion_matrix = confusion_matrix(y_test, y_pred)

        return self.metrics

    def print_metrics(self):
        """Metrikleri yazdırır"""
        print("\nModel Performans Metrikleri:")
        print("-" * 30)
        for metric, value in self.metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("\nDetaylı Sınıflandırma Raporu:")
        print("-" * 30)
        print(self.classification_report)

        print("\nKarmaşıklık Matrisi:")
        print("-" * 30)
        print(self.confusion_matrix)