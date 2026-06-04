from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("loan_pipeline.pkl")


def get_confidence(model, df):
    """
    Extract approval confidence from the pipeline.
    Works with soft VotingClassifier (predict_proba available).
    Falls back gracefully if voting='hard' or proba not supported.
    """
    try:
        proba = model.predict_proba(df)[0]
        # proba[1] = probability of class 1 (Approved)
        # proba[0] = probability of class 0 (Rejected)
        approved_proba = float(proba[1])
        rejected_proba = float(proba[0])
        return round(approved_proba * 100, 1), round(rejected_proba * 100, 1)
    except AttributeError:
        # Hard voting — no probabilities available
        return None, None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test")
def test():
    sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban"
    }
    df = pd.DataFrame([sample])
    pred = model.predict(df)[0]
    approved_conf, rejected_conf = get_confidence(model, df)
    return (
        f"Prediction: {'Approved' if pred == 1 else 'Rejected'} | "
        f"Confidence → Approved: {approved_conf}%  Rejected: {rejected_conf}%"
    )


@app.route("/predict_form", methods=["POST"])
def predict_form():
    data = {
        "Gender":            request.form["Gender"],
        "Married":           request.form["Married"],
        "Dependents":        request.form["Dependents"],
        "Education":         request.form["Education"],
        "Self_Employed":     request.form["Self_Employed"],
        "ApplicantIncome":   float(request.form["ApplicantIncome"]),
        "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
        "LoanAmount":        float(request.form["LoanAmount"]),
        "Loan_Amount_Term":  float(request.form["Loan_Amount_Term"]),
        "Credit_History":    float(request.form["Credit_History"]),
        "Property_Area":     request.form["Property_Area"],
    }

    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    approved_conf, rejected_conf = get_confidence(model, df)
    is_approved = (pred == 1)

    # Confidence shown is for the predicted outcome
    display_confidence = approved_conf if is_approved else rejected_conf

    return render_template(
        "result.html",
        approved=is_approved,
        confidence=display_confidence,          # e.g. 87.4  (for predicted class)
        approved_conf=approved_conf,            # e.g. 87.4
        rejected_conf=rejected_conf,            # e.g. 12.6
        has_proba=(approved_conf is not None),
        applicant_income=request.form["ApplicantIncome"],
        loan_amount=request.form["LoanAmount"],
        loan_term=request.form["Loan_Amount_Term"],
        credit_history=request.form["Credit_History"],
        property_area=request.form["Property_Area"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        approved_conf, rejected_conf = get_confidence(model, df)
        return jsonify({
            "prediction":     int(pred),
            "status":         "Approved" if pred == 1 else "Rejected",
            "approved_prob":  approved_conf,
            "rejected_prob":  rejected_conf,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)