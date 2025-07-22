from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re
from waitress import serve  # ✅ only once!

app = Flask(__name__)
CORS(app)

# Load ML model and vectorizer
clf = joblib.load('../model/model.pkl')
vectorizer = joblib.load('../model/vectorizer.pkl')

# Load phone specs
phone_specs = pd.read_csv('../data/phone_specs.csv', encoding='utf-8-sig')
phone_specs.columns = phone_specs.columns.str.strip()
phone_specs['camera_quality'] = phone_specs['camera_quality'].astype(int)
phone_specs['battery_life'] = phone_specs['battery_life'].astype(int)
phone_specs['performance'] = phone_specs['performance'].astype(int)
phone_specs['price'] = phone_specs['price'].astype(int)
phone_specs['ip_rating'] = phone_specs['ip_rating'].fillna('none')

# Recognized feature keywords
FEATURE_KEYWORDS = ['camera', 'battery', 'performance', 'ip', 'gaming', 'rugged']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    query = data.get('query', '').lower().strip()

    if not query:
        return jsonify({'error': 'Query is empty'}), 400

    if not any(kw in query for kw in FEATURE_KEYWORDS):
        return jsonify({
            'message': "Tell me what kind of phone you want: camera, battery, gaming, or rugged?"
        })

    # Predict relevant feature to sort by
    category = clf.predict(vectorizer.transform([query]))[0].lower()

    sort_col = {
        'camera_phone': 'camera_quality',
        'battery_phone': 'battery_life',
        'gaming_phone': 'performance',
        'ip_phone': 'ip_rating'
    }.get(category, 'price')  # fallback

    df = phone_specs.copy()

    # Extract price cap
    nums = re.findall(r'\d+', query)
    if nums:
        max_price = int(nums[0])
        df = df[df['price'] <= max_price]

    # Sort results
    df = df.sort_values(by=[sort_col, 'price'], ascending=[False, True])

    # Top 5
    top_phones = df.head(5)[[
        'name', 'camera_quality', 'battery_life', 'performance', 'ip_rating', 'price'
    ]].to_dict(orient='records')

    return jsonify({
        'category': category,
        'recommendations': top_phones
    })

# ✅ Serve using Waitress
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8000)
