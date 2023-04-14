#!/usr/bin/env python
# from onmt.online_translator import RecognizerParameter, ASROnlineTranslator
from onmt.online_translator import TranslatorParameter, OnlineTranslator
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

#if len(sys.argv)<=2:
filename = "model.conf"
print(host, port)
#else:
#    filename = sys.argv[3]

# I have no idea what these lines are doing
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
    """
    This function is used in checking if the prefixes and languages are the same or not
    Args:
        my_list:

    Returns:

    """
    my_list = list(set(my_list))
    return my_list


def initialize_model():
    """
    Build the translator
    """
    model = OnlineTranslator(filename)
    print("MT Model initialized")

    max_batch_size = 16

    return model, max_batch_size

def use_model(reqs):

    if len(reqs) == 1:
        req = reqs[0]
        input_text, prefix, input_language, output_language = req.get_data()
        model.set_language(input_language, output_language)
        hypo = model.translate(input_text, [prefix])
        result = {"hypo": hypo}
        req.publish(result)

    else:
        input_texts = list()
        prefixes = list()
        input_languages = list()
        output_languages = list()

        batch_runnable = False

        for req in reqs:
            input_text, prefix, input_language, output_language = req.get_data()
            model.set_language(input_language, output_language)
            input_texts.append(input_text)
            prefixes.append(prefix)

            input_languages.append(input_language)
            output_languages.append(output_language)

        unique_prefix_list = create_unique_list(prefixes)
        unique_input_languages = create_unique_list(input_languages)
        unique_output_languages = create_unique_list(output_languages)

        if len(unique_prefix_list) == 1 and len(unique_input_languages) == 1 and len(unique_output_languages) == 1:
            batch_runnable = True

        if batch_runnable:
            model.set_language(input_languages[0], output_languages[0])
            hypos = model.translate_batch(input_texts, prefixes)

            for req, hypo in zip(reqs, hypos):
                result = {"hypo": hypo}
                req.publish(result)
        else:
            for req, input_text, prefix, input_language, output_language \
                    in zip(reqs, input_texts, prefixes, input_languages, output_languages):
                model.set_language(input_language, output_language)

                hypo = model.translate(input_text, [prefix])
                result = {"hypo": hypo}
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


# corresponds to an asr_server "http://$host:$port/asr/infer/en,en" in StreamASR.py
# use None when no input- or output language should be specified
# @app.route("/asr/infer/<input_language>,<output_language>", methods=["POST"])
@app.route("/predictions/<input_language>,<output_language>", methods=["POST"])
def inference(input_language, output_language):
    # pcm_s16le: bytes = request.files.get("pcm_s16le").read()
    # prefix = request.files.get("prefix") # can be None
    # if prefix is not None:
    #     prefix: str = prefix.read().decode("utf-8")

    # note: in ASR/SLT it should be "request.files"
    # while in MT it's "request.data"

    input_text = request.form['text']  # can be None

    try:
        prefix = request.form['prefix']    # can be None
    except:
        prefix = None

    #
    print("RECEIVED INPUT TEXT:", input_text)

    try:
        priority = request.form["priority"]  # can be None
        priority = int(priority.read()) # used together with priority queue
    except:
        priority = 0

    condition = threading.Condition()
    with condition:
        id = str(uuid.uuid4())

        # the same with SLT
        data = (input_text, prefix, input_language, output_language)

        queue_in.put(Priority( priority, id, condition, data))

        condition.wait()

    result = dict_out.pop(id)
    status = 200
    if status in result:
        status = result.pop(status)

    # result has to contain a key "hypo" with a string as value (other optional keys are possible)
    return json.dumps(result), status

# called during automatic evaluation of the pipeline to store worker information
@app.route("/models/<input_language>,<output_language>", methods=["GET"])
def version(input_language, output_language):
    # print(input_language, output_language)
    # return dict or string (as first argument)
    return conf_data, 200

model, max_batch_size = initialize_model()

queue_in = queue.PriorityQueue()
dict_out = {}

decoding = threading.Thread(target=run_decoding)
decoding.daemon = True
decoding.start()

app.run(host=host, port=port)
