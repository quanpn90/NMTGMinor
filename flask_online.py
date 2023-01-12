from onmt.online_translator import RecognizerParameter, ASROnlineTranslator
from flask import Flask, request
import torch
import numpy as np
import math
import sys
import json

host = sys.argv[1]  # 192.168.0.72
port = sys.argv[2]  # 5051

app = Flask(__name__)

filename = "model.conf"

model = ASROnlineTranslator(filename)
print("ASR initialized")


def pcm_s16le_to_tensor(pcm_s16le):
    audio_tensor = np.frombuffer(pcm_s16le, dtype=np.int16)
    audio_tensor = torch.from_numpy(audio_tensor)
    audio_tensor = audio_tensor.float() / math.pow(2, 15)
    audio_tensor = audio_tensor.unsqueeze(1)  # shape: frames x 1 (1 channel)
    return audio_tensor


# corresponds to an asr_server "http://$host:$port/asr/infer/en,en" in StreamASR.py
# use None when no input- or output language should be specified
@app.route("/asr/infer/<input_language>,<output_language>", methods=["POST"])
def inference(input_language, output_language):
    pcm_s16le: bytes = request.files.get("pcm_s16le").read()
    prefix = request.files.get("prefix")  # can be None
    if prefix is not None:
        prefix: str = prefix.read().decode("utf-8")

    # calculate features corresponding to a torchaudio.load(filepath) call
    audio_tensor = pcm_s16le_to_tensor(pcm_s16le)

    try:
        if prefix is not None:
            prefix = [prefix]
        hypo = model.translate(audio_tensor, prefix)

        result = {"hypo": hypo}
    except Exception as e:
        print(e)
        return "An error occured", 400

    # result has to contain a key "hypo" with a string as value (other optional keys are possible)
    return json.dumps(result), 200


# called during automatic evaluation of the pipeline to store worker information
@app.route("/asr/version", methods=["POST"])
def version():
    return "ASR-7EU-1.0", 200


app.run(host=host, port=port)
