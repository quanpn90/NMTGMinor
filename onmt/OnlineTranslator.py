import onmt
import onmt.modules


class TranslatorParameter(object):

    def __init__(self,filename):

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
        self.gpu = -1
        self.cuda = False
        self.verbose = False
        self.alpha = 0.6
        self.beta = 0.0
        self.start_with_bos = False
        self.fp16 = False
        self.stop_early = True
        self.normalize = False
        self.len_penalty = 0.6
        self.bos_token = "BOS"
        self.ensemble_op = "logSum"
        self.diverse_beam_strength = 0.5
        self.bos_token = '#en#'
        self.start_with_tag = ''
        self.force_target_length = False
        
        self.readFile(filename)

    def readFile(self,filename):

        f = open(filename)

        line = f.readline()

        while(line):

            w = line.strip().split()

            if (w[0] == 'model'):
                self.model = w[1]
            elif (w[0] == 'beam_size'):
                self.beam_size = int(w[1])
            elif (w[0] == 'gpu'):
                self.gpu = int(w[1])
            elif (w[0] == 'cuda'):
                self.cuda = w[1].lower() == 'true'
            elif (w[0] == 'bos_token'):
                self.bos_token = w[1]

            line = f.readline()


class OnlineTranslator(object):
    def __init__(self,model):
        opt = TranslatorParameter(model)
        self.translator = onmt.EnsembleTranslator(opt)
    

    def translate(self,input):
              output = self.translator.translate([input.split()],[])
              translations = output[0][0]
              probabilities = output[1][0]
              return ' '.join(translations[0]), probabilities[0]

