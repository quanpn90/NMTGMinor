import onmt.markdown
import argparse


parser = argparse.ArgumentParser(description='plm_dict.py')

onmt.markdown.add_md_help_argument(parser)
parser.add_argument('-plm_src_vocab', required=True,
                    help="Path to the pretrained model src vocab data")
parser.add_argument('-src_lang', required=True,
                    help="src language")

parser.add_argument('-plm_tgt_vocab', required=True,
                    help="Path to the pretrained model tgt vocab data")
parser.add_argument('-tgt_lang', required=True,
                    help="tgt language")

parser.add_argument('-plm_type', default="bert", type=str,
                    help=""" the type of pretrained language trained model""")

opt = parser.parse_args()


def load_vocab(vocab_file, lang):
    """Loads a vocabulary file into a dictionary."""
    index = 0
    vocab = open(vocab_file, "r")
    word2idx = open(opt.plm_type + "_word2idx."+lang, "w")
    idx2word = open(opt.plm_type + "_idx2word."+lang, "w")
    while True:
        word = vocab.readline()
        # the last line
        if not word:
            break
        word = word.strip()
        idx2word.write(str(index) + " " + word + "\n")
        word2idx.write(word + " " + str(index) + "\n")
        index += 1

    vocab.close()
    word2idx.close()
    idx2word.close()


load_vocab(opt.plm_src_vocab, opt.src_lang)
load_vocab(opt.plm_tgt_vocab, opt.tgt_lang)

