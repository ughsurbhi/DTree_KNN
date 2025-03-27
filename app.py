from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)


dt_model = joblib.load("decision_tree_model.pkl")
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)  # Reshape for model
        features_scaled = scaler.transform(features)  # Apply StandardScaler

        dt_prediction = dt_model.predict(features)[0]
        knn_prediction = knn_model.predict(features_scaled)[0]

        iris_classes = ["setosa", "versicolor", "virginica"]
        dt_species = iris_classes[dt_prediction]
        knn_species = iris_classes[knn_prediction]

        return jsonify({
            "Decision Tree Prediction": dt_species,
            "KNN Prediction": knn_species
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
