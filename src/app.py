from model.predict_model import make_prediction
from flask import Flask, render_template, request
from data.vocab import Vocab

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def translate():
    input = request.form.get("modern")
    trans = make_prediction("mod_a.ckpt", input)
    return render_template("index.html", sample_output=trans)

if __name__ == "__main__":
    app.run(debug=True)