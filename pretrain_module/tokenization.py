import onmt.markdown
import argparse

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.markdown.add_md_help_argument(parser)

parser.add_argument('-data_file', default="",
                    help="Path to the data")
parser.add_argument('-plm_vocab_file', default="", type=str,
                    help="""Path of vocab file""")
parser.add_argument('-lang', default='en',
                    help='language [en|ch|others]')
parser.add_argument('-plm_type', default='roberta',
                    help='pretrain_mode [roberta|bert|others]')
parser.add_argument('-pretrain_tokenizer', default='roberta-base',
                    help='which tokenizer is used')
parser.add_argument('-add_special_tok', action='store_true',
                    help="""add special tokens at the beginning and at the end of each sentence""")
parser.add_argument('-special_bos', default='<s>',
                    help='special bos token.')
parser.add_argument('-special_eos', default='</s>',
                    help='special eso token.')
opt = parser.parse_args()


def make_vocab_dict(vocab_file, lang):
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


def tokenize_data(raw_data, tokenizer):
    with open(raw_data, "r", encoding="utf-8") as f_raw:
        tokenized_sents = []
        for line in f_raw:
            sent = line.strip()
            tokenized_sent = tokenizer.tokenize(sent)
            if opt.add_special_tok:
                tokenized_sent.insert(0, opt.special_bos)
                tokenized_sent.append(opt.special_eos)

            tokenized_sents.append(tokenized_sent)

    new_data = raw_data + "." + opt.plm_type + ".tok"
    with open(new_data, "w", encoding="utf-8") as f_tok:
        for sent in tokenized_sents:
            sent = " ".join(sent)
            f_tok.write(sent)
            f_tok.write('\n')


def main():

    # step1: make dictionary
    make_vocab_dict(opt.plm_vocab_file, opt.lang)

    # step2: tokenization
    if opt.plm_type == "bert":
        from pytorch_pretrained_bert import BertTokenizer
        # "en": bert-base-uncased  "ch": bert-base-chinese
        tokenizer = BertTokenizer.from_pretrained(opt.pretrain_tokenizer)
    elif opt.plm_type == "roberta":
        from pretrain_module.roberta_tokenization_ch import FullTokenizer
        from transformers import RobertaTokenizer
        # "en": roberta-base
        if opt.lang != "ch":
            tokenizer = RobertaTokenizer.from_pretrained(opt.pretrain_tokenizer)
        else:
            tokenizer = FullTokenizer(opt.plm_vocab_file)
    else:
        print("Tokenization with this pretrained model is not supported right now.")
        exit(-1)

    if opt.data_file != "":
        tokenize_data(opt.data_file, tokenizer)


if __name__ == "__main__":
    main()
