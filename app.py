from flask import Flask, request, abort
from datetime import datetime
import os
from src.preprocessing import image_proprocessing
from src.prediction import predict
import json

app = Flask(__name__)

UPLOADER_FOLDER = "request_images"


@app.route("/image", methods=["POST"])
def get_image():
    image = request.files.get("image", "")

    try:
        filename = f"{datetime.now().isoformat()}.jpg"

        image.save(os.path.join(UPLOADER_FOLDER, filename))

        image_batch = image_proprocessing(filename)

        labels = predict(image_batch)

        return json.dumps(labels)

    except Exception as ex:
        print("pizdec nahui")

        print(repr(ex))

        abort(400)


def main():
    app.run(
        port=7836
    )


if __name__ == "__main__":
    main()