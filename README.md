# 🤖 AI-Powered Loan Eligibility Advisor

<img src="static/loan.png" alt="Project Banner Image" width="100%">

* * *

## 📖 Introduction

**AI-Powered Loan Eligibility Advisor** is a modern full-stack web application that predicts loan eligibility using a **machine learning model** (Decision Tree Classifier) and an **AI-powered chatbot** (Google Gemini).

Users can:

*   Create an account
    
*   Access protected pages
    
*   Fill financial details via a smart form
    
*   Or use the conversational chatbot for guided predictions
    

This project is a professional, secure, Flask-based upgrade of the original Streamlit version — redesigned for real-world use, scalability, and much better UI/UX.

* * *

## ✨ Key Features

*   🔐 **User Authentication** (Firebase)
    
*   🔒 **Protected Routes & Token Verification**
    
*   📋 **Loan Prediction Form**
    
*   💬 **AI Chatbot Advisor** (Google Gemini)
    
*   🧠 **Machine Learning Model** (`model.pkl`)
    
*   🎨 **Responsive UI** with HTML + CSS + JS
    
*   ⚙️ **Firebase Admin SDK**
    
*   🌐 **Flask Backend API**
    

* * *

## 🛠️ Tech Stack

### Backend

*   Flask
    
*   Firebase Admin SDK
    
*   Google Gemini API
    
*   Scikit-learn
    
*   Pandas / NumPy
    
*   python-dotenv
    

### Frontend

*   HTML5
    
*   CSS3
    
*   JavaScript
    
*   Firebase Authentication (Client SDK)
    

* * *

# 🚀 Getting Started

Follow these steps to run the project locally.

* * *

## 1️⃣ Clone the Project

`git clone https://github.com/your-username/ai-powered_loan_eligibility_advisor.git cd ai-powered_loan_eligibility_advisor`

* * *

## 2️⃣ Project Structure

'ai-powered_loan_eligibility_advisor/
├── Flask_app.py
├── model.pkl
├── serviceAccountKey.json
├── requirements.txt
├── .env
├── static/
│   ├── css/styles.css
│   ├── js/
│   │   ├── firebase_auth.js
│   │   ├── predict_form.js
│   │   └── chatbot.js
│   └── images/
├── templates/
│   ├── layout.html
│   ├── login.html
│   ├── index.html
│   ├── about.html
│   ├── predict.html
│   └── chatbot.html
└── README.md'


* * *

## 3️⃣ Set Up the Python Environment

### Create and activate a virtual environment:

#### Windows

`python -m venv venv venv\Scripts\activate`

#### macOS / Linux

`python3 -m venv venv source venv/bin/activate`

### requirements.txt

Create a file named **requirements.txt** and paste:

`flask numpy google-generativeai pandas joblib scikit-learn firebase-admin python-dotenv`

Install dependencies:

`pip install -r requirements.txt`

* * *

## 4️⃣ Configure Firebase

Firebase has **two parts**: server-side (Flask) and client-side (frontend).

* * *

### A. Server-Side Setup

1.  Go to Firebase Console → Project Settings → **Service accounts**
    
2.  Click **Generate new private key**
    
3.  Rename the file:
    

`serviceAccountKey.json`

4.  Place it inside your project:
    

`ai-powered_loan_eligibility_advisor/serviceAccountKey.json`

* * *

### B. Client-Side Setup

1.  Go to Firebase Console → Project Settings → **General**
    
2.  Scroll to **Your apps**
    
3.  Create a **Web App**
    
4.  Copy the `firebaseConfig`
    

Open:

`templates/layout.html`

Replace the placeholder Firebase config with your own.

* * *

## 5️⃣ Configure Google Gemini

### Steps:

1.  Open **Google AI Studio**
    
2.  Create an API key
    
3.  Create a `.env` file:
    

`GEMINI_API_KEY=YOUR_API_KEY_HERE`

* * *

## 6️⃣ Run the Application

`flask run`

Now visit:

👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

* * *

# 📈 Machine Learning Model — `model.pkl`

The model was trained in **practice.ipynb** using **train.csv**.

### Training steps:

*   Load & clean dataset
    
*   Handle missing values
    
*   Encode categorical variables
    
*   Log-transform skewed numerical features
    
*   Train a `DecisionTreeClassifier`
    
*   Save model as `model.pkl`
    

### Model used in:

*   ✔ Loan Prediction Form
    
*   ✔ AI Chatbot Advisor
    

* * *

# 🏛️ System Architecture

### Chatbot Flow

1.  User logs in
    
2.  Chatbot asks 11 financial questions
    
3.  Flask preprocesses inputs
    
4.  `model.pkl` predicts **Yes / No**
    
5.  Gemini generates an explanation
    
6.  User receives final recommendation
    

<img src="static/system\_architecture.png" width="100%">

* * *

# 🖼️ Screenshots

| Home Page | About Page |
| --- | --- |
| <img src="static/loan.png" width="400"> | <img src="static/Project Activity.png" width="400"> |

| Prediction Form | Chatbot |
| --- | --- |
| (Add screenshot here) | (Add screenshot here) |

* * *

# 📧 Contact

**Developed by:** **Jishan**  
📩 Email: **jishan2305@gmail.com**  
🌐 GitHub: **your-github-profile**
