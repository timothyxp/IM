from flask import Flask, request, abort
from datetime import datetime
import os
from src.preprocessing import image_proprocessing
from src.prediction import predict
import json
import pandas as pd

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

        data = pd.read_csv("imagenet.csv")

        result = []

        for label in labels:
            res = {"class_name": label[0], "tag": label[1], "probability": label[2]}
            row = data[data.class_id == label[0]].loc[0]
            res["artist"] = row.artist
            res["song_name"] = row.music_name
            res["music_img"] = row.music_img
            res["youtube_url"] = row.youtube_link

            result.append(res)

        return json.dumps(result)

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