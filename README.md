# 📱 Phone Finder AI — Flask + ML Backend

An AI-powered smartphone recommendation system built using Flask, scikit-learn, and Waitress. It takes natural language queries from users like _"Best camera phone under 1 lakh"_ and returns personalized smartphone suggestions based on their needs.

---

## 🚀 Features

- 🔍 Understands natural language queries
- 📊 Uses machine learning (Logistic Regression + TF-IDF)
- 📱 Recommends top 5 phones based on features (camera, battery, performance, ruggedness)
- 🌐 Provides a REST API (`/predict`) for frontend consumption
- 🧠 Pre-trained ML model (model.pkl) with custom vectorizer (vectorizer.pkl)
- ⚙️ Production-ready using Waitress WSGI server

---

## 🛠️ Technologies Used

| Technology        | Purpose                                  |
|-------------------|------------------------------------------|
| Flask             | Web API framework                        |
| scikit-learn      | Machine Learning model training          |
| pandas            | CSV handling and data processing         |
| TfidfVectorizer   | Text feature extraction from user query  |
| LogisticRegression| Query classification                     |
| Waitress          | Production-grade Python WSGI server      |
| Flask-CORS        | Handles CORS for frontend API calls      |
| JavaScript (AJAX) | Used in frontend to communicate with API |

---

git clone https://github.com/yourusername/phonefinder-backend.git
cd phonefinder-backend

pip install flask flask_cors waitress pandas scikit-learn joblib

python model/train_model.py

python api/app.py

open index.html with live server (Now you can chat with this model)
