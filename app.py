from flask import Flask, request, abort, jsonify
from datetime import datetime
import os
from src.preprocessing import image_proprocessing
from src.prediction import predict
from src.helpers import fix_na
import json
import pandas as pd

app = Flask(__name__)

UPLOADER_FOLDER = "request_images"

DEBUG = False


@app.route("/image", methods=["POST"])
def get_image():
    print(dir(request))

    print(request.files)

    print(request.files.get("photo"))
    image = request.files.get("photo", "")

    print(dir(image))

    try:
        if DEBUG:
            filename = os.path.join(UPLOADER_FOLDER, "2019-10-26T22:13:18.942507.jpg")
        else:
            filename = os.path.join(UPLOADER_FOLDER, f"{datetime.now().isoformat()}.jpg")

            image.save(filename)

        print("save completed")

        image_batch = image_proprocessing(filename)

        print("start predicting")
        labels = predict(image_batch)

        print("reading csv")
        data = pd.read_csv("imagenet.csv").fillna("")

        result = []

        songs = set()

        for label in labels:
            print(label)
            res = {"class_name": label[0], "tag": label[1], "probability": float(label[2])}
            row = data[data.class_id == label[0]].iloc[0]
            print(row)
            res["artist"] = fix_na(row.artist)
            res["song_name"] = fix_na(row.music_name)
            res["music_img"] = fix_na(row.music_img)
            res["youtube_url"] = fix_na(row.youtube_link)

            if row.music_name not in songs and row.artist.replace(" ", "") != "":
                songs.add(row.music_name)
                result.append(res)

        if len(result) == 0:
            result.append({
                "class_name": "",
                "tag": "",
                "probability": "",
                "artist": "Imagine Dragons",
                "song_name": "I'm So Sorry",
                "youtube_url": "https://www.youtube.com/watch?v=D0gddT_DDzQ",
                "music_img": "https://i.ytimg.com/vi/8dWDD8P71Is/maxresdefault.jpg"
            })

        return jsonify(result)

    except Exception as ex:
        print("pizdec nahui")

        print(repr(ex))

        abort(400)


def main():
    os.makedirs(UPLOADER_FOLDER, exist_ok=True)
    app.run(
        port=7836
    )


if __name__ == "__main__":
    main()