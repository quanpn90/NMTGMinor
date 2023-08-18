#!/usr/bin/env python
from onmt.online_translator import RecognizerParameter, ASROnlineTranslator
from flask import Flask, request
import torch
import numpy as np
import math
import sys
import json
import threading
import queue
import uuid
import traceback
import subprocess

host = sys.argv[1]  # 192.168.0.72
port = sys.argv[2]  # 5051
if len(sys.argv)<=3:
    filename = "model.conf"
else:
    filename = sys.argv[3]

conf_data = open(filename,"r").read().split("\n")
model = None
for d in conf_data:
    d = d.split()
    if len(d)==2 and d[0]=="model":
        model = d[1]
        break
conf_data.append("model_ls "+str(subprocess.run(("ls -l "+model).split(), capture_output=True).stdout))
conf_data = "\n".join(conf_data)

app = Flask(__name__)

def create_unique_list(my_list):
    my_list = list(set(my_list))
    return my_list

def initialize_model():
    model = ASROnlineTranslator(filename)
    print("ASR initialized")

    max_batch_size = 16

    return model, max_batch_size

def use_model(reqs):

    if len(reqs) == 1:
        req = reqs[0]
        audio_tensor, prefix, input_language, output_language, memory = req.get_data()
        model.set_language(input_language, output_language)
        hypo, bpe_output, scores = model.translate(audio_tensor, [prefix], memory)
        result = {"hypo": hypo, "bpe_hypo": bpe_output, "log_probs": scores}
        req.publish(result)

    else:
        audio_tensors = list()
        prefixes = list()
        input_languages = list()
        output_languages = list()
        memories = list()

        batch_runnable = False

        for req in reqs:
            audio_tensor, prefix, input_language, output_language, memory = req.get_data()
            model.set_language(input_language, output_language)
            audio_tensors.append(audio_tensor)
            prefixes.append(prefix)

            input_languages.append(input_language)
            output_languages.append(output_language)
            memories.append(memory)

        unique_prefix_list = create_unique_list(prefixes)
        unique_input_languages = create_unique_list(input_languages)
        unique_output_languages = create_unique_list(output_languages)
        memories = create_unique_list(memories)

        if len(unique_prefix_list) == 1 and len(unique_input_languages) == 1 and len(unique_output_languages) == 1 and len(memories) == 1:
            batch_runnable = True

        if batch_runnable:
            model.set_language(input_languages[0], output_languages[0])
            hypos, bpe_outputs, all_scores = model.translate_batch(audio_tensors, prefixes, memories[0])

            for req, hypo, bpe_output, scores in zip(reqs, hypos, bpe_outputs, all_scores):
                result = {"hypo": hypo, "bpe_hypo": bpe_output, "log_probs": scores}
                req.publish(result)
        else:
            for req, audio_tensor, prefix, input_language, output_language, memory \
                    in zip(reqs, audio_tensors, prefixes, input_languages, output_languages, memories):
                model.set_language(input_language, output_language)

                hypo, bpe_output, scores = model.translate(audio_tensor, [prefix], memory)
                result = {"hypo": hypo, "bpe_hypo": bpe_output, "log_probs": scores}
                req.publish(result)

def run_decoding():
    while True:
        reqs = [queue_in.get()]
        while not queue_in.empty() and len(reqs) < max_batch_size:
            req = queue_in.get()
            reqs.append(req)
            if req.priority >= 1:
                break

        print("Batch size:",len(reqs),"Queue size:",queue_in.qsize())

        try:
            use_model(reqs)
        except Exception as e:
            print("An error occured during model inference")
            traceback.print_exc()
            for req in reqs:
                req.publish({"hypo":"", "status":400})

class Priority:
    next_index = 0

    def __init__(self, priority, id, condition, data):
        self.index = Priority.next_index

        Priority.next_index += 1

        self.priority = priority
        self.id = id
        self.condition = condition
        self.data = data

    def __lt__(self, other):
        return (-self.priority, self.index) < (-other.priority, other.index)

    def get_data(self):
        return self.data

    def publish(self, result):
        dict_out[self.id] = result
        try:
            with self.condition:
                self.condition.notify()
        except:
            print("ERROR: Count not publish result")

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
    prefix = request.files.get("prefix") # can be None
    if prefix is not None:
        prefix: str = prefix.read().decode("utf-8")
    memory = request.files.get("memory") # can be None
    if memory is not None:
        memory: list = json.loads(memory.read())

    # calculate features corresponding to a torchaudio.load(filepath) call
    audio_tensor = pcm_s16le_to_tensor(pcm_s16le)

    priority = request.files.get("priority") # can be None
    try:
        priority = int(priority.read()) # used together with priority queue
    except:
        priority = 0

    condition = threading.Condition()
    with condition:
        id = str(uuid.uuid4())
        data = (audio_tensor,prefix,input_language,output_language,memory)

        queue_in.put(Priority(priority,id,condition,data))

        condition.wait()

    result = dict_out.pop(id)
    status = 200
    if status in result:
        status = result.pop(status)

    # result has to contain a key "hypo" with a string as value (other optional keys are possible)
    return json.dumps(result), status

# called during automatic evaluation of the pipeline to store worker information
@app.route("/asr/version", methods=["POST"])
def version():
    # return dict or string (as first argument)
    return conf_data, 200

model, max_batch_size = initialize_model()

queue_in = queue.PriorityQueue()
dict_out = {}

decoding = threading.Thread(target=run_decoding)
decoding.daemon = True
decoding.start()

app.run(host=host, port=port)
