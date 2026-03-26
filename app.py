import os
import tempfile
from flask import Flask, request, jsonify, render_template
from src.prediction.ASD_prediction import ASD_Prediction

app = Flask(__name__)

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = ASD_Prediction()
    return pipeline


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    fname = file.filename
    suffix = ".nii.gz" if fname.endswith(".nii.gz") else os.path.splitext(fname)[1]

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.close(fd)
        file.save(tmp_path)

        pl = get_pipeline()
        pred, prob, _ = pl.predict(tmp_path)

        autism_prob = float(prob[list(pl.model.classes_).index("autism")])
        label = str(pred).lower()

        return jsonify({
            "label":       "Autistic Sample" if label == "autism" else "Non-Autistic Sample",
            "prediction":  label,
            "probability": round(autism_prob, 8),
            "percent":     int(autism_prob * 100),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=7860)