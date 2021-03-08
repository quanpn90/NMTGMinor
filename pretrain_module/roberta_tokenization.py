import onmt.markdown
import argparse
from pretrain_module.roberta_tokenization_ch import FullTokenizer
from transformers import RobertaTokenizer

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.markdown.add_md_help_argument(parser)

parser.add_argument('-src_data', default="",
                    help="Path to the source data")
parser.add_argument('-tgt_data', default="",
                    help="Path to the target data")

parser.add_argument('-vocab_file_tgt', default="", type=str,
                    help="""Path of vocab file""")

parser.add_argument('-src_lang', default='en',
                    help='Source language')
parser.add_argument('-tgt_lang', default='zh',
                    help='Target language')

opt = parser.parse_args()


def tokenize_data(raw_data, tokenizer, lang):
    with open(raw_data, "r", encoding="utf-8") as f_raw:
        tokenized_sents = []
        for line in f_raw:
            sent = line.strip()
            tokenized_sent = tokenizer.tokenize(sent)
            if lang == "en":
                tokenized_sent.insert(0, "<s>")
                tokenized_sent.append("</s>")
            elif lang == "zh":
                tokenized_sent = tokenized_sent

            tokenized_sents.append(tokenized_sent)

    new_data = raw_data + ".roberta.tok"
    with open(new_data, "w", encoding="utf-8") as f_tok:
        for sent in tokenized_sents:
            sent = " ".join(sent)
            f_tok.write(sent)
            f_tok.write('\n')


def main():
    src_lang = opt.src_lang
    tgt_lang = opt.tgt_lang

    if src_lang == "en":
        tokenizer_src = RobertaTokenizer.from_pretrained('roberta-base')

    if tgt_lang == "zh":
        tokenizer_tgt = FullTokenizer(opt.vocab_file_tgt)

    if opt.src_data is not "": 
        tokenize_data(opt.src_data, tokenizer_src, src_lang)
    if opt.tgt_data is not "": 
        tokenize_data(opt.tgt_data, tokenizer_tgt, tgt_lang)


if __name__ == "__main__":
    main()
