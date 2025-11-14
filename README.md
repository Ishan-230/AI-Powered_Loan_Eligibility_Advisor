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

### 3\. Set Up the Python Environment

Create and activate a virtual environment:

Bash

    # Windows
    python -m venv venv
    venv\Scripts\activate
    
    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate

Create a file named `requirements.txt` and paste the following content into it:

# requirements.txt
    flask
    numpy
    google-generativeai
    pandas
    joblib
    scikit-learn
    firebase-admin
    python-dotenv

Now, install all the dependencies:

Bash

    pip install -r requirements.txt

### 4\. Configure Firebase

There are two Firebase configurations you must complete.

**A. Server-Side Key (for Flask)**

1.  Go to your Firebase Console -> Project Settings -> **Service accounts**.
    
2.  Click **"Generate new private key"** and download the JSON file.
    
3.  **Rename** this file to `serviceAccountKey.json`.
    
4.  Place it in the root of your project folder (at the same level as `Flask_app.py`).
    

**B. Client-Side Config (for the UI)**

1.  Go to your Firebase Console -> Project Settings -> **General** tab.
    
2.  Scroll down to "Your apps" and select (or create) a **Web App** (</>).
    
3.  Find the `firebaseConfig` object.
    
4.  Open `templates/layout.html` and paste your `firebaseConfig` object into the script at the bottom of the page, replacing the placeholder.
    

### 5\. Configure Google Gemini

1.  Go to [Google AI Studio](https://ai.google.dev/) and get your API key.
    
2.  In the root of your project, create a file named `.env`.
    
3.  Add your API key to this file:
    
    GEMINI_API_KEY=YOUR_API_KEY_HERE
    

### 6\. Run the Application

You're all set! Run the Flask app:

Bash

    flask run

Open your browser and go to `http://127.0.0.1:5000` to see the application.

* * *

## 📈 The Machine Learning Model

The heart of this project is the `model.pkl` file, which is a **Decision Tree Classifier** model trained on the `train.csv` dataset.

The model was built in the `practice.ipynb` notebook. The process involved:

1.  **Data Loading:** Importing the `train.csv` dataset.
    
2.  **Data Cleaning:** Handling missing values (`NaN`) in columns like `Gender`, `Married`, `Credit_History`, etc.
    
3.  **Feature Engineering:** Converting categorical data (like `Gender`, `Property_Area`) into numerical values.
    
4.  **Log Transformation:** Applying `np.log` to skewed data like `ApplicantIncome` and `LoanAmount` to normalize their distribution.
    
5.  **Training:** Training a `DecisionTreeClassifier` on the cleaned, processed features to predict the `Loan_Status` column.
    
6.  **Exporting:** Saving the trained model as `model.pkl` using `pickle`.
    

This model is loaded by the Flask app and is used by **both** the prediction form and the chatbot to generate the final 'Y' or 'N' prediction.

* * *

## 🏛️ System Architecture

The application uses a hybrid AI approach. The process flow for the chatbot is as follows:

1.  A user logs in via Firebase Authentication.
    
2.  The user answers 11 questions from the chatbot.
    
3.  The Flask server collects all 11 answers.
    
4.  The server runs the `preprocess_data` function and feeds the results into the `model.pkl` (Decision Tree) to get a prediction (`Y` or `N`).
    
5.  The server then sends the user's data _and_ the model's prediction to the **Google Gemini API**, asking it to write a user-friendly, formatted response.
    
6.  The final, nicely-formatted advice is sent back to the user's chat window.
    

<img src="static/system\_architecture.png" alt="System Architecture Diagram" width="100%">

* * *

## 🖼️ Screenshots

| Home Page | About Page |
| --- | --- |
| <img src="static/loan.png" alt="Home Page" width="400"> | <img src="static/Project Activity.png" alt="Project Activity Log" width="400"> |
| Prediction Form | Chatbot |
| --- | --- |
| (Add a screenshot of your prediction form here) | (Add a screenshot of your chatbot interface here) |

* * *

## 📧 Contact

This project was developed by Jishan.

*   **Email:** [jishan2305@gmail.com](mailto:jishan2305@gmail.com)
    
*   GitHub: your-github-profile
    
    (Replace with your GitHub profile link)
