import onmt
import onmt.modules


class TranslatorParameter(object):

    def __init__(self, filename):

        self.model = "";
        self.src = "<stdin>";
        self.src_img_dir = "";
        self.tgt = "";
        self.output = "<stdout>";
        self.beam_size = 1
        self.batch_size = 1
        self.max_sent_length = 100
        self.dump_beam = ""
        self.n_best = self.beam_size
        self.replace_unk = False
        self.gpu = -1;
        self.cuda = 0;
        self.verbose = False

        self.beta=0.0
        self.alpha=0.0
        self.start_with_bos=True
        self.fp16=False
        self.ensemble_op='mean'
        self.autoencoder=None
        self.encoder_type='text'
        
        self.read_file(filename)

    def read_file(self,filename):

        f = open(filename)

        line = f.readline()

        while line:

            w = line.strip().split()

            if w[0] == "model":
                self.model = w[1]
            elif w[0] == "beam_size":
                self.beam_size = int(w[1])

            line = f.readline()


class OnlineTranslator(object):
    def __init__(self, model):
        opt = TranslatorParameter(model)
        self.translator = onmt.EnsembleTranslator(opt)
    

    def translate(self,input):
        predBatch, predScore, predLength, goldScore, numGoldWords,allGoldScores = self.translator.translate([input.split()],[])

        return " ".join(predBatch[0][0])
  

