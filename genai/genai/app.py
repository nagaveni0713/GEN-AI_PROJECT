from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load Pre-trained BERT Model
MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
print("Loading BERT Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
print("BERT Model loaded successfully.")

def predict_sentiment(text):
    try:
        encoded_text = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        output = model(**encoded_text)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        
        # Labels: 0->1 star, 1->2 stars, 2->3 stars, 3->4 stars, 4->5 stars
        star_rating = np.argmax(scores) + 1
        confidence = np.max(scores)
        
        if star_rating <= 2:
            sentiment = 'Negative'
        elif star_rating == 3:
            sentiment = 'Neutral'
        else:
            sentiment = 'Positive'
            
        return sentiment, confidence, star_rating
    except Exception as e:
        print(f"Error: {e}")
        return "Neutral", 0.0, 3

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    confidence = None
    polarity = None
    review_text = ""

    if request.method == 'POST':
        review_text = request.form['review']
        if review_text.strip():
            sentiment, conf, stars = predict_sentiment(review_text)
            confidence = f"{conf*100:.2f}%"
            # Map stars to a polarity-like score for display
            # 1 star -> -1.0, 3 stars -> 0.0, 5 stars -> 1.0
            polarity = (stars - 3) / 2.0 

    return render_template('index.html', 
                           sentiment=sentiment, 
                           confidence=confidence, 
                           review=review_text,
                           score=polarity)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
