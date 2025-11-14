# 🤖 AI-Powered Loan Eligibility Advisor

<img src="static/loan.png" alt="Project Banner Image" width="100%">

### **[Live Demo](https://your-live-app-url.com)**
*(Replace with your deployed app's link)*

---

## 📖 Introduction

This project is a modern, full-stack web application designed to predict loan eligibility. It combines a **machine learning model** (a Decision Tree trained on historical loan data) with a **conversational AI chatbot** (powered by Google Gemini) to create a seamless, user-friendly experience.

Users can create a secure account, submit their financial details through a simple form, or be guided through the process by an intelligent chatbot. The app provides instant predictions and helpful, actionable advice.

This Flask application is an enhanced version of an original Streamlit prototype, rebuilt with a focus on professional UI/UX, security, and scalability.

---

## ✨ Key Features

* **Secure User Authentication:** Full login and registration system using **Firebase Authentication**.
* **Protected Routes:** The Prediction Form and AI Chatbot pages are accessible only to logged-in users.
* **Dual Prediction Methods:**
    1.  **Prediction Form:** A clean, multi-column form for quick, direct eligibility checks.
    2.  **AI Chatbot Advisor:** A conversational, step-by-step guide that collects user data, submits it, and presents the results in a human-friendly way.
* **Hybrid AI Model:** Uses **Google Gemini** for natural language understanding and **your `model.pkl`** (a trained Decision Tree) to perform the final, accurate prediction.
* **Modern, Responsive UI:** Built with clean HTML, CSS, and JavaScript for a professional look that works on both desktop and mobile.

---

## 🛠️ Tech Stack

* **Backend:**
    * [Flask](https://flask.palletsprojects.com/) (Python Web Framework)
    * [Google Gemini](https://ai.google.dev/) (For AI Chatbot conversation)
    * [Firebase Admin SDK](https://firebase.google.com/docs/admin/setup) (For server-side token verification)
* **Frontend:**
    * HTML5
    * CSS3 (with modern layouts)
    * JavaScript (for auth and API calls)
    * [Firebase Auth SDK](https://firebase.google.com/docs/auth/web/start) (For client-side login/register)
* **Machine Learning:**
    * [Scikit-learn](https://scikit-learn.org/stable/) (For the Decision Tree model)
    * [Pandas](https://pandas.pydata.org/) (For data preprocessing)
    * [NumPy](https://numpy.org/) (For numerical operations)
    * [Jupyter Notebook](https://jupyter.org/) (For model training and experimentation)

---

## 🚀 How to Run This Project Locally

Follow these steps to set up and run the project on your own machine.

### 1. Prerequisites
* Python 3.10 or newer
* A [Google Firebase](https://console.firebase.google.com/) project
* A [Google AI Studio](https://ai.google.dev/) API Key (for Gemini)

### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/ai-powered_loan_eligibility_advisor.git](https://github.com/your-username/ai-powered_loan_eligibility_advisor.git)
cd ai-powered_loan_eligibility_advisor
