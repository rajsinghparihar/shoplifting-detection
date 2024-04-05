from flask import Flask, request, jsonify
from api import DetectionModel
import os

app = Flask(__name__)
model_instance = DetectionModel()
temp_filepath = "./temp/temp.mp4"
if not os.path.exists("./temp/"):
    os.makedirs("./temp/")


@app.route("/", methods=["GET"])
def index():
    return """<!DOCTYPE html>
  <html>
  <body>
    <a href='http://127.0.0.1:5000/streamlit'>Go to prediction app</a>
  </body>
  </html>"""


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded!"})
    video_fs = request.files["video"]
    video_fs.save(temp_filepath)
    output_path = model_instance.detect_shoplifting(temp_filepath)

    # Get video info
    video_info = model_instance._utils.get_video_info(output_path)

    return jsonify({"output_path": output_path, "video_info": video_info})


if __name__ == "__main__":
    app.run(debug=True)
