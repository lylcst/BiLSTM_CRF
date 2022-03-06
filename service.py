from predict import predict
from flask import Flask, jsonify
import flask

app = Flask(__name__)


@app.route('/service/api/medical_ner', methods=["POST", "GET"])
def index():
    data = {"success": 0}
    text_list = flask.request.get_json()["text_list"]
    items = []
    for text in text_list:
        res = predict(text)
        items.append(res)
    data["data"] = items
    data["success"] = 1

    return jsonify(data)


if __name__ == '__main__':
    app.run(port=8089)

