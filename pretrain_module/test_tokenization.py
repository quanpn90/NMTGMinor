from tokenization_mbart50_clusters import MBart50ClusterTokenizer

# tokenizer = MBart50ClusterTokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX")

vocab_file = "sentencepiece/sentencepiece.bpe.model"

tokenizer = MBart50ClusterTokenizer(vocab_file=vocab_file,
                                    src_lang=None,
                                    tgt_lang=None,
                                    tokenizer_file=None,
                                    eos_token="</s>",
                                    sep_token="</s>",
                                    cls_token="<s>",
                                    unk_token="<unk>",
                                    pad_token="<pad>",
                                    mask_token="<mask>", )

print(tokenizer)

src_text = "<s> UN Chief Says <s> en_XX There Is No Military Solution in Syria"

src = tokenizer(src_text)

print(src)

src_text = "__14__ __14__ __14__ __14__ __14__ __131__ __131__ __131__ __131__ __131__ __131__ __131__ __131__ __188__"

src = tokenizer(src_text)

print(src)