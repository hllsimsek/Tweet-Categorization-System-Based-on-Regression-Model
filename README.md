# Kategorix Platform

## 🇬🇧 English Description

A dynamic tweet classification and filtering platform that allows users to view categorized tweets under topics like politics, sports, health, technology, and lifestyle.

### 🌐 Technologies Used

- **HTML5 & CSS3** – for structure and styling  
- **Bootstrap 5** – responsive design and UI elements  
- **JavaScript** – dynamic UI interactions and filters  
- **Font Awesome** – icons for categories  
- **Animate.css** – animations and transitions  
- **Flask (Python)** – API backend  
- **JSON** – for tweet data storage  
- **scikit-learn** – for machine learning  
- **Gensim** – for Word2Vec embeddings  
- **NLTK & re** – for Turkish tweet preprocessing and cleaning

### 🤖 Machine Learning Model

The core of the system is a **logistic regression** classifier trained on Turkish-language tweets.  
To enhance model performance:

- **TF-IDF** (Term Frequency–Inverse Document Frequency) was used to represent tweet texts as numerical vectors.  
- Additionally, **Word2Vec** embeddings were explored to capture semantic relationships.  
- Each tweet was labeled with categories such as *Politics, Sports, Technology, Health, Lifestyle*, and used in supervised training.

### 🚀 Features

- Displays tweets categorized by topic
- Shows tweet text, author (`@username`), number of likes, and publish date
- Allows users to filter tweets by **date** or **like count**
- Each category has a unique color theme
- Smooth UI animations and mobile responsiveness

### 🧠 Project Purpose

To make browsing social content easier by organizing posts under relevant categories and enhancing user experience with filtering and UI enhancements.

---

## 🇹🇷 Türkçe Açıklama

Kategorix, kullanıcıların tweet içeriklerini **siyaset, spor, sağlık, teknoloji ve yaşam** gibi başlıklar altında görmesini sağlayan dinamik bir tweet sınıflandırma ve filtreleme platformudur.

### 🌐 Kullanılan Teknolojiler

- **HTML5 & CSS3** – yapı ve tasarım  
- **Bootstrap 5** – responsive arayüz ve bileşenler  
- **JavaScript** – dinamik filtreleme ve kullanıcı etkileşimleri  
- **Font Awesome** – kategori ikonları  
- **Animate.css** – animasyonlar ve geçişler  
- **Flask (Python)** – API katmanı  
- **JSON** – tweet verisi saklama  
- **scikit-learn** – makine öğrenmesi işlemleri  
- **Gensim** – Word2Vec vektörleri  
- **NLTK & re** – Türkçe tweet ön işleme ve temizleme

### 🤖 Makine Öğrenmesi Modeli

Sistem, Türkçe tweet'ler üzerinde eğitilmiş bir **lojistik regresyon** sınıflandırıcısı kullanır.  
Model başarımını artırmak için:

- Tweet metinleri sayısal vektörlere dönüştürmek amacıyla **TF-IDF** yöntemi kullanılmıştır.  
- Ayrıca, semantik ilişkileri yakalamak adına **Word2Vec** gömme (embedding) yöntemi de uygulanmıştır.  
- Her tweet, *Siyaset, Spor, Teknoloji, Sağlık, Yaşam* gibi etiketlerle sınıflandırılmış ve denetimli öğrenme (supervised learning) ile eğitilmiştir.

### 🚀 Özellikler

- Tweet’leri kategori bazlı listeler  
- Tweet metni, kullanıcı adı (`@username`), beğeni sayısı ve paylaşım tarihi gösterilir  
- Kullanıcı, tweet’leri **tarihe** veya **beğeni sayısına** göre filtreleyebilir  
- Her kategoriye özel renk teması  
- Akıcı kullanıcı arayüzü ve mobil uyumluluk

### 🧠 Proje Amacı
Tweet içeriklerini karışık şekilde sunmak yerine ilgili başlıklar altında düzenleyerek, kullanıcı deneyimini filtreleme ve görsel iyileştirmelerle daha kullanışlı hale getirmeyi amaçlar.
