import os
from onmt.OnlineTranslator import TranslatorParameter,OnlineTranslator
import sys


filename="/model/model.conf"

t = OnlineTranslator(filename)
print "NMT initialized";
sys.stdout.flush()

while True:
#    sys.stderr.write("Waiting for data\n");
    line = sys.stdin.readline();
#    sys.stderr.write("Input: "+line+"\n");
    print t.translate(line);
#    sys.stderr.write("Translation done\n");
    sys.stdout.flush()
