from flask import Flask, render_template, jsonify, request
import json

app = Flask(__name__, template_folder='templates')

def load_categorized_tweets():
    try:
        with open('categorized_tweets.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading tweets: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tweets')
def api_tweets():
    tweets = load_categorized_tweets()
    sort_by = request.args.get('sort', 'date')  # date veya likes
    order = request.args.get('order', 'desc')  # asc veya desc
    
    # Kategori eşleştirme
    category_map = {
        'Yaşam': 'yasam',
        'Spor': 'spor',
        'Sağlık': 'saglik',
        'Siyaset': 'siyaset',
        'Teknoloji': 'teknoloji'
    }
    
    # Tweet'leri kategorilere göre grupla
    categorized = {
        'yasam': [],
        'spor': [],
        'saglik': [],
        'siyaset': [],
        'teknoloji': []
    }
    
    for tweet in tweets:
        category = tweet.get('category')
        if category in category_map:
            mapped_category = category_map[category]
            # Tweet verilerini işle
            tweet_data = {
                'text': tweet['text'],
                'username': tweet.get('username', 'Bilinmeyen Kullanıcı'),
                'date': tweet.get('date', ''),
                'likes': tweet.get('likes', '0')
            }
            categorized[mapped_category].append(tweet_data)
    
    # Her kategori için tweetleri sırala
    for category in categorized:
        if sort_by == 'date':
            categorized[category].sort(
                key=lambda x: str(x.get('date', '')),  # Tarihi string olarak sırala
                reverse=(order == 'desc')
            )
        elif sort_by == 'likes':
            categorized[category].sort(
                key=lambda x: int(x.get('likes', '0')),  # Beğenileri integer olarak sırala
                reverse=(order == 'desc')
            )
    
    return jsonify(categorized)

if __name__ == '__main__':
    app.run(debug=True)
