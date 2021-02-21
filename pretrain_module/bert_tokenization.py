import sys
sys.path.append('/home/dhe/hiwi/Exercises/NMTGMinor/')
from pytorch_pretrained_bert import BertTokenizer
import onmt.markdown
import argparse

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.markdown.add_md_help_argument(parser)

parser.add_argument('-src_data',  default="",
                    help="Path to the source data")
parser.add_argument('-tgt_data', default="",
                    help="Path to the target data")

opt = parser.parse_args()


def tokenize_data(raw_data, tokenizer, lang):
    with open(raw_data, "r", encoding="utf-8") as f_raw:
        tokenized_sents = []
        for line in f_raw:
            sent = line.strip()
            if lang == "en":
                marked_sent = "[CLS] " + sent + " [SEP]"

            # In data preprocessing the BOS and EOS tokens are added in tgt sentences
            elif lang == "zh":
                marked_sent = sent
            tokenized_sent = tokenizer.tokenize(marked_sent)
            tokenized_sents.append(tokenized_sent)

    new_data = raw_data + ".bert.tok"
    with open(new_data, "w", encoding="utf-8") as f_tok:
        for sent in tokenized_sents:
            sent = " ".join(sent)
            f_tok.write(sent)
            f_tok.write('\n')


def main():
    tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_zh = BertTokenizer.from_pretrained('bert-base-chinese')

    src_lang = "en"
    tgt_lang = "zh"
    
    if opt.src_data != "":
        tokenize_data(opt.src_data, tokenizer_en, src_lang)

    if opt.tgt_data != "":
        tokenize_data(opt.tgt_data, tokenizer_zh, tgt_lang)


if __name__ == "__main__":
    main()
