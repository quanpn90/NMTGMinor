import os
from onmt.online_translator import RecognizerParameter, ASROnlineTranslator
from onmt.utils import safe_readaudio
import sys
from flask import Flask, request
#
# app = Flask(__name__)
#
# @app.route("/asr/infer", methods=["GET"])
# def inference():
#     prefix = request.form.get("prefix")
#     raw_audio = request.form.get("raw_audio")
#     try:
#         result = model.inference(raw_audio, prefix) # TODO: provide such an interface
#     except:
#         return "An error occured", 400
#     return result, 200
#
# filename="/model/model.conf"
#
# model = ASROnlineTranslator(filename)
# print("NMT initialized")
# sys.stdout.flush() # TODO: load model
#
# app.run(debug=True, host="0.0.0.0", port=5051, use_reloader=True)




filename="model.conf"

t = ASROnlineTranslator(filename)
print("ASR initialized")
sys.stdout.flush()

# Testing time
data = open("test.seg")
line = data.readline()

# print(line)

"/export/data1/chuber/ASR/data/EN/tedlium/wav/ZubaidaBai_2016S.wav 105.5 114.63 " \
"My mind was flooded with reflections of my own infection that I had to struggle with for a year past childbirth, despite having access to the best medical care."
wav_path = "/export/data1/chuber/ASR/data/EN/tedlium/wav/ZubaidaBai_2016S.wav"
start = 105.5
end = 114.63
line = safe_readaudio(wav_path, start=start, end=end, sample_rate=16000)

src_length = line.size(0)
input = line

prefix = ["My mind was flooded with reactions"]

output = t.translate(input, prefix)


print(output)