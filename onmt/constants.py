import torch

PAD = 0
UNK = 1
BOS = 2
EOS = 3

# for Bert, both en and zh; also for roberta zh
BERT_PAD = 0
BERT_UNK = 100
BERT_BOS = 101
BERT_EOS = 102
BERT_MASK = 103


# for Roberta_en
EN_ROBERTA_PAD = 1
EN_ROBERTA_UNK = 3
EN_ROBERTA_BOS = 0
EN_ROBERTA_EOS = 2


MASK_WORD = '[MASK]'
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

