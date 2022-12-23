import os
from onmt.online_translator import RecognizerParameter, ASROnlineTranslator
from onmt.utils import safe_readaudio
import sys
from flask import Flask, request
import torch
import numpy as np
import math
import sys

host = sys.argv[1]  # 192.168.0.72
port = sys.argv[2]  # 5051

app = Flask(__name__)

filename = "model.conf"

model = ASROnlineTranslator(filename)
print("ASR initialized")


@app.route("/asr/en/infer", methods=["GET"])
def inference():
    prefix = request.files.get("prefix")
    audio_bytes = request.files.get("raw_audio").read()

    if prefix is not None:
        prefix = prefix.read().decode("utf-8")
    # tensor = safe_readaudio(raw_audio_file, sample_rate=16000)
    # breakpoint()
    tensor = np.frombuffer(audio_bytes, dtype=np.int16)
    tensor = torch.from_numpy(tensor)
    # normalize
    tensor = tensor.float() / math.pow(2, 15)
    tensor = tensor.unsqueeze(1)

    if prefix is not None:
        prefix = [prefix]
    try:
        result = model.translate(tensor, prefix)  # TODO: provide such an interface
    except Exception as e:
        print(e)
        return "An error occured", 400
    return result, 200



app.run(debug=True, host=host, port=port, use_reloader=True)