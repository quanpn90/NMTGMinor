import sys
sys.path.append('/home/dhe/hiwi/Exercises/NMTGMinor')

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

opt = parser.parse_args()



vocab_file="/project/student_projects2/dhe/BERT/experiments/pytorch_pretrained_models/roberta-base-layer12-zh/bert-base-chinese-vocab.txt"

def tokenize_data(raw_data, tokenizer, lang):
    with open(raw_data, "r", encoding="utf-8") as f_raw:
        tokenized_sents = []
        for line in f_raw:
            sent = line.strip()

            tokenized_sent = tokenizer.tokenize(sent)

            # 注意特殊符号前后空格
            # tgt(zh) 在preprocess 文件中有加开始和结束符号，所以不用在这里加了
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
    tokenizer_zh = FullTokenizer(vocab_file)
    # wordpiece_tokenizer_zh 只是整个tokenization中的一部分， 我们用端到端的tokenizer_zh
    # wordpiece_tokenizer_zh = tokenizer_zh.wordpiece_tokenizer
    tokenizer_en = RobertaTokenizer.from_pretrained('roberta-base')

    src_lang = "en"
    tgt_lang = "zh"

    if opt.src_data is not "": 
        print("tokenzize src data")
        tokenize_data(opt.src_data, tokenizer_en, src_lang)
    if opt.tgt_data is not "": 
        print("tokenzize tgt data")
        tokenize_data(opt.tgt_data, tokenizer_zh, tgt_lang)


if __name__ == "__main__":
    main()
