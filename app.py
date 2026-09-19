from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from Inference import DEFAULT_MODEL, DEFAULT_THRESHOLD, TBPredictor

ROOT = Path(__file__).resolve().parent
WEB_OUTPUTS = ROOT / "web_outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
WEB_OUTPUTS.mkdir(parents=True, exist_ok=True)
predictor = TBPredictor(DEFAULT_MODEL)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def web_file(path):
    return f"/files/{path.name}"


@app.route("/files/<path:filename>")
def files(filename):
    return send_from_directory(WEB_OUTPUTS, filename)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            error = "Choose a chest X-ray image first."
        elif not allowed_file(uploaded.filename):
            error = "Use PNG, JPG, JPEG, BMP, TIF, or TIFF images."
        else:
            job_id = uuid4().hex[:10]
            extension = Path(secure_filename(uploaded.filename)).suffix.lower()
            input_path = WEB_OUTPUTS / f"{job_id}_input{extension}"
            overlay_path = WEB_OUTPUTS / f"{job_id}_overlay.png"
            uploaded.save(input_path)
            try:
                result = predictor.predict(
                    input_path,
                    threshold=DEFAULT_THRESHOLD,
                    heatmap_path=overlay_path,
                )
                result["input_url"] = web_file(input_path)
                result["overlay_url"] = web_file(overlay_path)
                result["heatmap_url"] = web_file(overlay_path.with_name(overlay_path.stem + "_heatmap.png"))
                result["report_url"] = web_file(overlay_path.with_name(overlay_path.stem + "_report.png"))
                result["device"] = str(predictor.device)
            except Exception as exc:
                input_path.unlink(missing_ok=True)
                error = f"Could not process this image: {exc}"
    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
