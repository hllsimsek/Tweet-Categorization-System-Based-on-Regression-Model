from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import json
from datetime import datetime, timedelta
import os

def load_existing_tweets():
    try:
        if os.path.exists('tweets.json'):
            with open('tweets.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except:
        return {}

def save_tweets(all_tweets):
    existing_tweets = load_existing_tweets()
    
    # Mevcut tweetleri güncelle/ekle
    for username, tweets in all_tweets.items():
        if username not in existing_tweets:
            existing_tweets[username] = []
        
        # Yeni tweetleri ekle (tekrar etmeyenleri)
        for new_tweet in tweets:
            # Tweet'in daha önce eklenip eklenmediğini kontrol et
            tweet_exists = any(
                existing_tweet['text'] == new_tweet['text'] and
                existing_tweet['date'] == new_tweet['date']
                for existing_tweet in existing_tweets[username]
            )
            
            if not tweet_exists:
                existing_tweets[username].append(new_tweet)
    
    # JSON dosyasına kaydet
    with open('tweets.json', 'w', encoding='utf-8') as f:
        json.dump(existing_tweets, f, ensure_ascii=False, indent=4)

def get_tweets(usernames, tweet_limit=50):
    # Chrome ayarları
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Tarayıcıyı arka planda çalıştır
    
    driver = webdriver.Chrome(options=chrome_options)
    all_tweets = {}

    for username in usernames:
        try:
            # Kullanıcının profiline git
            driver.get(f'https://twitter.com/{username}')
            time.sleep(3)  # Sayfanın yüklenmesini bekle
            
            tweets = []
            last_height = driver.execute_script("return document.body.scrollHeight")
            
            # Son 24 saatin tarihini hesapla
            one_day_ago = datetime.now() - timedelta(days=1)
            
            while len(tweets) < tweet_limit:
                # Tweet elementlerini bul
                tweet_elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                
                for tweet in tweet_elements:
                    if len(tweets) >= tweet_limit:
                        break
                        
                    try:
                        # Tweet metnini al
                        tweet_text = tweet.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]').text
                        
                        # Tweet tarihini al
                        time_element = tweet.find_element(By.TAG_NAME, 'time')
                        tweet_date = time_element.get_attribute('datetime')
                        
                        # Beğeni sayısını al
                        try:
                            like_element = tweet.find_element(By.CSS_SELECTOR, '[data-testid="like"]')
                            like_count = like_element.text
                            if not like_count:  # Eğer beğeni yoksa '0' olarak ayarla
                                like_count = '0'
                        except:
                            like_count = '0'
                        
                        # Kullanıcı adını al
                        try:
                            username_element = tweet.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
                            display_name = username_element.text.split('\n')[0]  # İlk satır görünen isim
                            user_handle = username_element.text.split('\n')[1]   # İkinci satır kullanıcı adı (@ile)
                        except:
                            display_name = username
                            user_handle = f'@{username}'
                        
                        # Tweet tarihini kontrol et
                        tweet_datetime = datetime.strptime(tweet_date, '%Y-%m-%dT%H:%M:%S.%fZ')
                        
                        # Sadece son 24 saatin tweetlerini ekle
                        if tweet_datetime > one_day_ago:
                            tweets.append({
                                'text': tweet_text,
                                'date': tweet_date,
                                'likes': like_count,
                                'display_name': display_name,
                                'username': user_handle
                            })
                    except:
                        continue
                
                # Sayfayı aşağı kaydır
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            
            all_tweets[username] = tweets
            
        except Exception as e:
            print(f"Hata oluştu ({username}): {str(e)}")
            continue
    
    driver.quit()
    return all_tweets

# Örnek kullanım
if __name__ == "__main__":
    # Takip edilecek kullanıcı listesi
    users_to_follow = ["cnnturk", "trtspor","SportsDigitale","trthaber","Haberturk",
                       "saglikbakanligi","360saglik","teknolojivbilim","miateknoloji",
                       "anadoluajansi","sabah","NASA","TCSanayi","Teknoloji","_samiyenhaber","NTV_Saglik"]
    # Tweet'leri çek
    all_tweets = get_tweets(users_to_follow, tweet_limit=25)
    
    # Yeni tweetleri mevcut JSON dosyasına ekle
    save_tweets(all_tweets)
    
    print("Tweet'ler başarıyla çekildi ve kaydedildi!")
