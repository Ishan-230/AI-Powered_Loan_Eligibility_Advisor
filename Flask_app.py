from flask import Flask, request, render_template, redirect, url_for, session, jsonify, g
from markupsafe import escape
import pickle
import numpy as np
import google.generativeai as genai
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
import os
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps
from dotenv import load_dotenv
load_dotenv() # Loads .env file for API key

app = Flask(__name__)

# --- Initialize Firebase Admin ---
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
except FileNotFoundError:
    print("WARNING: serviceAccountKey.json not found. Firebase Admin SDK not initialized.")
except ValueError:
    pass # Already initialized

# --- Set a secret key for Flask sessions ---
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# --- Load Model ---
model = pickle.load(open("model.pkl", 'rb'))

# --- Load API Key from .env ---
gemini_api_key = os.environ.get('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please check your .env file.")
genai.configure(api_key=gemini_api_key)

# --- Define the Gemini model (NO tools/function calling) ---
# We will use this model *after* we have our prediction.
gemini_model = genai.GenerativeModel(model_name='gemini-2.5-flash-lite')


# --- Chatbot Questions ---
questions = [
    "What is your gender? (Male/Female)",
    "Are you married? (Yes/No)",
    "How many dependents do you have? (0/1/2/3+)",
    "What is your education level? (Graduate/Not Graduate)",
    "Are you self-employed? (Yes/No)",
    "What is your monthly applicant income?",
    "What is your monthly co-applicant income?",
    "What is the loan amount you are requesting?",
    "What is the loan term in days?",
    "What is your credit history score? (300-1000)",
    "What is the property area? (Urban/Semiurban/Rural)"
]

# --- Preprocessing Function (Synced with Streamlit logic + Bug Fixes) ---
def preprocess_data(gender, married, dependents, education, employed, credit, area,
                   ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
    try:
        male = 1 if gender.lower() == "male" else 0
        married_yes = 1 if married.lower() == "yes" else 0
        
        # Handles '3' and '3+'
        if dependents == '1':
            dependents_1, dependents_2, dependents_3 = 1, 0, 0
        elif dependents == '2':
            dependents_1, dependents_2, dependents_3 = 0, 1, 0
        elif dependents == "3+" or dependents == "3": # <-- Bug Fix
            dependents_1, dependents_2, dependents_3 = 0, 0, 1
        else: # This correctly handles '0'
            dependents_1, dependents_2, dependents_3 = 0, 0, 0

        not_graduate = 1 if education.lower() == "not graduate" else 0
        employed_yes = 1 if employed.lower() == "yes" else 0
        semiurban = 1 if area.lower() == "semiurban" else 0
        urban = 1 if area.lower() == "urban" else 0

        # BUG FIX: Handle log(0)
        ApplicantIncomelog = np.log(float(ApplicantIncome) if float(ApplicantIncome) > 0 else 1)
        total_income = float(ApplicantIncome) + float(CoapplicantIncome)
        totalincomelog = np.log(total_income if total_income > 0 else 1)
        LoanAmountlog = np.log(float(LoanAmount) if float(LoanAmount) > 0 else 1)
        Loan_Amount_Termlog = np.log(float(Loan_Amount_Term) if float(Loan_Amount_Term) > 0 else 1)
        
        # LOGIC SYNC: Use >= 800 (from Streamlit)
        if float(credit) >= 800 and float(credit) <= 1000:
            credit_binary = 1
        else:
            credit_binary = 0

        return [
            credit_binary, ApplicantIncomelog, LoanAmountlog, Loan_Amount_Termlog, totalincomelog,
            male, married_yes, dependents_1, dependents_2, dependents_3, not_graduate, employed_yes, semiurban, urban
        ]
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None


# --- Login Decorator (for API routes) ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        id_token = None
        if 'Authorization' in request.headers:
            id_token = request.headers['Authorization'].split(' ').pop()

        if not id_token:
            return jsonify({"message": "Authentication token is missing!"}), 401

        try:
            decoded_token = auth.verify_id_token(id_token)
            g.user_id = decoded_token['uid']
        except Exception as e:
            return jsonify({"message": f"Token verification failed: {str(e)}"}), 401

        return f(*args, **kwargs)
    return decorated_function

# --- Standard Routes ---

@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/prediction')
def prediction_form():
    return render_template("index.html") # Page is locked via JS in the template

@app.route('/chatbot')
def chatbot():
    session['current_step'] = -1
    session['responses'] = {}
    return render_template("chatbot.html") # Page is locked via JS in the template

# --- Auth Routes ---

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# --- Prediction and Chat API Routes ---

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        # Form submission from index.html
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        employed = request.form['employed']
        credit = float(request.form['credit']) # 0/1 value from form JS
        area = request.form['area']
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])

        # Preprocessing logic
        male = 1 if (gender == "Male") else 0
        married_yes = 1 if (married == "Yes") else 0
        
        if (dependents == '1'):
            dependents_1, dependents_2, dependents_3 = 1, 0, 0
        elif dependents == '2':
            dependents_1, dependents_2, dependents_3 = 0, 1, 0
        elif dependents == '3+' or dependents == '3': # <-- Bug Fix
            dependents_1, dependents_2, dependents_3 = 0, 0, 1
        else:
            dependents_1, dependents_2, dependents_3 = 0, 0, 0
            
        not_graduate = 1 if education == "Not Graduate" else 0
        employed_yes = 1 if (employed == "Yes") else 0
        if area == "Semiurban":
            semiurban, urban = 1, 0
        elif area == "Urban":
            semiurban, urban = 0, 1
        else:
            semiurban, urban = 0, 0

        ApplicantIncomeLog = np.log(ApplicantIncome if ApplicantIncome > 0 else 1)
        total_income = ApplicantIncome + CoapplicantIncome
        totalincomelog = np.log(total_income if total_income > 0 else 1)
        LoanAmountLog = np.log(LoanAmount if LoanAmount > 0 else 1)
        Loan_Amount_Termlog = np.log(Loan_Amount_Term if Loan_Amount_Term > 0 else 1)

        features = [[credit, ApplicantIncomeLog, LoanAmountLog, Loan_Amount_Termlog, totalincomelog, male, married_yes,
                     dependents_1, dependents_2, dependents_3, not_graduate, employed_yes, semiurban, urban]]
        
        prediction = model.predict(features)
        
        prediction_text = "Loan status is Yes" if (prediction == "Y") else "Loan status is No"
        
        return render_template("prediction.html", prediction_text=prediction_text)
    
    # If GET request, just show the form
    return render_template("index.html")


@app.route('/chat_api', methods=['POST'])
@login_required # API is protected
def chat_api():
    try:
        data = request.json
        user_input = data['message']
        
        current_step = session.get('current_step', -1)
        responses = session.get('responses', {})
        
        if user_input == '__INIT__':
            session['current_step'] = -1
            session['responses'] = {}
            return jsonify({'response': "Hello! I'm your AI Loan Advisor. To check your eligibility, I need to ask you 11 questions. Are you ready to begin? (Yes/No)"})

        if current_step == -1:
            if user_input.lower() == "yes":
                session['current_step'] = 0
                bot_response = "Great! Let's get started.\n\n" + questions[0]
            else:
                bot_response = "No problem! Just type 'Yes' when you're ready to begin."
            session['responses'] = {}
            return jsonify({'response': bot_response})

        elif current_step < len(questions):
            # Use string keys for session dictionary
            responses[str(current_step)] = user_input
            session['current_step'] = current_step + 1
            
            if session['current_step'] < len(questions):
                bot_response = questions[session['current_step']]
                session['responses'] = responses
                session.modified = True
                return jsonify({'response': bot_response})
            
            # --- All questions are answered ---
            else:
                session['responses'] = responses
                session.modified = True
                
                try:
                    r = session['responses']
                    
                    # --- NEW LOGIC (from chatbot.py) ---
                    
                    # 1. Get all 11 responses
                    gender = r['0']
                    married = r['1']
                    dependents = r['2'] # Will be '3' or '3+'
                    education = r['3']
                    self_employed = r['4']
                    applicant_income = r['5']
                    coapplicant_income = r['6']
                    loan_amount = r['7']
                    loan_amount_term = r['8']
                    credit_history = r['9']
                    property_area = r['10']

                    # 2. Preprocess the data
                    features = preprocess_data(
                        gender, married, dependents, education, self_employed, 
                        credit_history, property_area, applicant_income, 
                        coapplicant_income, loan_amount, loan_amount_term
                    )
                    
                    if features is None:
                        raise ValueError("Data preprocessing failed.")

                    # 3. Use *your* model.pkl to predict
                    prediction = model.predict([features])
                    model_result = 'Eligible for loan' if prediction[0] == 'Y' else 'Not eligible for loan'

                    # 4. Create the final prompt for Gemini (copied from chatbot.py)
                    prompt = f"""
                    I want to check my eligibility for a loan. Here is my information:

                    Gender: {gender}
                    Marital Status: {married}
                    Dependents: {dependents}
                    Education: {education}
                    Self-Employed: {self_employed}
                    Applicant Income: {applicant_income}
                    Coapplicant Income: {coapplicant_income}
                    Loan Amount: {loan_amount}
                    Loan Amount Term: {loan_amount_term}
                    Credit History: {credit_history}
                    Property Area: {property_area}

                    The result of the prediction model is: '{model_result}'

                    Please evaluate the above details and the model result, and:

                    If I am 'Eligible for loan', provide the next steps to complete the loan process, such as:
                    - Documents I need to prepare.
                    - Steps to finalize the loan application.
                    - Estimated timeline for loan disbursement.
                    - Tips to maintain a good credit score during this process.

                    If I am 'Not eligible for loan', kindly suggest actionable steps to improve my eligibility. Specifically:
                    - Areas I should focus on (e.g., increasing income, improving credit history, reducing loan amount).
                    - Recommended credit score improvement strategies.
                    - How to enhance financial stability to meet the loan criteria in the future.
                    - Alternative financing options or smaller loans I might qualify for.

                    Provide your response in a structured and actionable format to help me take the next steps efficiently.
                    Start the response with a clear "Congratulations!" or "Based on your details..."
                    """
                    
                    # 5. Send the prompt to Gemini (no function calling)
                    response = gemini_model.generate_content(prompt)
                    bot_response = response.text

                    # 6. Reset session
                    session['current_step'] = -1
                    session['responses'] = {}
                    return jsonify({'response': bot_response})

                except Exception as e:
                    session['current_step'] = -1
                    session['responses'] = {}
                    return jsonify({'response': f"An error occurred while processing: {str(e)}. Please type 'Yes' to start over."})

        return jsonify({'response': "Something went wrong. Please type 'Yes' to start over."})

    except Exception as e:
        print(f"--- FATAL ERROR IN CHAT_API ---")
        print(str(e))
        print(f"---------------------------------")
        session['current_step'] = -1
        session['responses'] = {}
        return jsonify({'message': f"An error occurred: {str(e)}. Please type 'yes' to start over."}), 500


if __name__ == "__main__":
    app.run(debug=True)