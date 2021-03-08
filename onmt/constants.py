import torch

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

checkpointing = 0
static = False
residual_type = 'regular'
max_position_length = 8192
torch_version = float(torch.__version__[:3])
double_precision = False

neg_log_sigma1 = 0
neg_log_sigma2 = 4
prior_pi = 0.5


def add_tokenidx(opt, cons, dicts):

    # the src_pad_word, tgt_pad_word etc are by default the same as before
    # changed if we use roberta/bert
    cons.SRC_PAD_WORD = opt.src_pad_word
    cons.SRC_UNK_WORD = opt.src_unk_word
    cons.SRC_BOS_WORD = opt.src_bos_word
    cons.SRC_EOS_WORD = opt.src_eos_word

    cons.TGT_PAD_WORD = opt.tgt_pad_word
    cons.TGT_UNK_WORD = opt.tgt_unk_word
    cons.TGT_BOS_WORD = opt.tgt_bos_word
    cons.TGT_EOS_WORD = opt.tgt_eos_word

    # In bilingual case there are two languages ("src" and "tgt")
    # in the dictionary
    if 'src' in dicts and 'tgt' in dicts:
        src_dict = dicts['src']
        cons.SRC_PAD = src_dict.labelToIdx[opt.src_pad_word]
        cons.SRC_UNK = src_dict.labelToIdx[opt.src_unk_word]
        cons.SRC_BOS = src_dict.labelToIdx[opt.src_bos_word]
        cons.SRC_EOS = src_dict.labelToIdx[opt.src_eos_word]

        tgt_dict = dicts['tgt']
        cons.TGT_PAD = tgt_dict.labelToIdx[opt.tgt_pad_word]
        cons.TGT_UNK = tgt_dict.labelToIdx[opt.tgt_unk_word]
        cons.TGT_BOS = tgt_dict.labelToIdx[opt.tgt_bos_word]
        cons.TGT_EOS = tgt_dict.labelToIdx[opt.tgt_eos_word]

    # for speech recognition we don't have dicts['src']
    elif 'tgt' in dicts:

        src_dict = dicts['tgt']
        cons.SRC_PAD = src_dict.labelToIdx[opt.src_pad_word]
        cons.SRC_UNK = src_dict.labelToIdx[opt.src_unk_word]
        cons.SRC_BOS = src_dict.labelToIdx[opt.src_bos_word]
        cons.SRC_EOS = src_dict.labelToIdx[opt.src_eos_word]

        tgt_dict = dicts['tgt']
        cons.TGT_PAD = tgt_dict.labelToIdx[opt.tgt_pad_word]
        cons.TGT_UNK = tgt_dict.labelToIdx[opt.tgt_unk_word]
        cons.TGT_BOS = tgt_dict.labelToIdx[opt.tgt_bos_word]
        cons.TGT_EOS = tgt_dict.labelToIdx[opt.tgt_eos_word]

    else:
        raise NotImplementedError

    return cons

# # for Bert, both en and zh; also for roberta zh
# BERT_PAD = 0
# BERT_UNK = 100
# BERT_BOS = 101
# BERT_EOS = 102
# BERT_MASK = 103
#
#
# # for Roberta_en
# EN_ROBERTA_PAD = 1
# EN_ROBERTA_UNK = 3
# EN_ROBERTA_BOS = 0
# EN_ROBERTA_EOS = 2
#
#
# MASK_WORD = '[MASK]'
# PAD_WORD = '<blank>'
# UNK_WORD = '<unk>'
# BOS_WORD = '<s>'
# EOS_WORD = '</s>'