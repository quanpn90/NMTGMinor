import torch

# first test create a layer

from onmt.models.speech_recognizer.w2v_bert.w2vbert_attention import TorchSDPA, create_default_sdpa
from onmt.models.speech_recognizer.w2v_bert.w2vbert_attention import RelativePositionalEncoding, RelativePositionSDPA
from onmt.models.speech_recognizer.w2v_bert.w2vbert_multihead_attention import StandardMultiheadAttention
from onmt.models.speech_recognizer.w2v_bert.norm_order import TransformerNormOrder

# test_module = TorchSDPA()


model_dim = 16
num_encoder_attn_heads = 1
pos_encoder = None
max_seq_len = 2048

attn_dropout_p = 0.0

sdpa = create_default_sdpa(attn_dropout_p=attn_dropout_p)
rel_pos_encoding = RelativePositionalEncoding(
    model_dim,
    max_seq_len,
)

# sdpa = RelativePositionSDPA(
#     model_dim,
#     num_encoder_attn_heads,
#     rel_pos_encoding,
#     inner_sdpa=sdpa
# )

attn_module = StandardMultiheadAttention(
    model_dim,
    num_encoder_attn_heads,
    pos_encoder=pos_encoder,
    sdpa=sdpa
)

print(attn_module)

batch_size = 8
seq_len = 32

seqs = torch.randn(batch_size, seq_len, model_dim)
padding_mask = torch.zeros(batch_size, seq_len).bool()
key_padding_mask = torch.zeros(batch_size, seq_len).bool()

# maybe cuda everything first?

attn_out = attn_module(seqs, padding_mask, seqs, key_padding_mask, seqs)

# seqs: Tensor,  # why don't you call this "query"
# padding_mask,
# keys: Tensor,
# key_padding_mask,
# values: Tensor,
print(attn_out.sum())

print("Test Relative Attention successfully")

# Next: test creating a Transformer Encoder (or Conformer Encoder) layer
from onmt.models.speech_recognizer.w2v_bert.w2vbert_transformer_encoder import StandardTransformerEncoderLayer
from onmt.models.speech_recognizer.w2v_bert.w2vbert_ffn import StandardFeedForwardNetwork

ffn_layer = StandardFeedForwardNetwork(model_dim, 4 * model_dim, True, inner_activation=torch.nn.SiLU())

ffn_input = torch.randn(batch_size, seq_len, model_dim)
ffn_output = ffn_layer(ffn_input)

print("Test FFN succesfully")

layer = StandardTransformerEncoderLayer(attn_module,
                                        ffn_layer,
                                        dropout_p=0.1,
                                        norm_order=TransformerNormOrder.POST)

layer_input = torch.randn(batch_size, seq_len, model_dim)
layer_output, _padding_mask = layer(layer_input, padding_mask=padding_mask, self_attn_mask=key_padding_mask)

print(layer_output.sum())
print("Test Transformer Layer succesfully")

# Next: test creating convolution/conformer layer

from onmt.models.speech_recognizer.w2v_bert.w2vbert_convolution import ConformerConvolution

depthwise_kernel_size = 3

conv_layer = ConformerConvolution(model_dim,
                                  depthwise_kernel_size,
                                  causal_depthwise_conv=False,
                                  norm_type="batch_norm",
                                  depthwise_activation=torch.nn.SiLU())

conv_input = torch.randn(batch_size, seq_len, model_dim)
conv_output = conv_layer(conv_input, padding_mask=padding_mask)

print(conv_output.sum())
print("Test Convolution Layer succesfully")

#  next: test creating a conformer block
from onmt.models.speech_recognizer.w2v_bert.w2vbert_conformer import ConformerBlock

ffn_layer2 = StandardFeedForwardNetwork(model_dim, 4 * model_dim, True, inner_activation=torch.nn.SiLU())

conformer_block = ConformerBlock(ffn_layer,
                                 attn_module,
                                 conv_layer,
                                 ffn_layer2,
                                 dropout_p=0.1,
                                 layer_norm_factory=None,
                                 )

block_input = torch.randn(batch_size, seq_len, model_dim)

block_output, _padding_mask = conformer_block(block_input, padding_mask=padding_mask, self_attn_mask=key_padding_mask)

print(block_output.sum())
block_output.sum().backward()
print("Test Conformer Layer succesfully")

# next: test creating a conformer/shaw model
from onmt.models.speech_recognizer.w2v_bert.w2vbert_config import W2VBertConfig, w2vbert_test_config, conformer_shaw_test

config = conformer_shaw_test()

print(config)

from onmt.models.speech_recognizer.w2v_bert.w2vbert_builder import ConformerShawEncoderBuilder, create_conformer_shaw_model
# builder =
w2vbert_model = create_conformer_shaw_model(config)

print(w2vbert_model)

input_size = 80

block_input = torch.randn(batch_size, seq_len, input_size)
padding_mask = key_padding_mask = torch.zeros(batch_size, seq_len).bool()

w2vbert_model_output = w2vbert_model(block_input, padding_mask)[0]

print(w2vbert_model_output.sum())
w2vbert_model_output.sum().backward()

print("Test creating w2vbert model successfully")

from onmt.models.speech_recognizer.w2v_bert.w2vbert_config import conformer_shaw_600m

config = conformer_shaw_600m()
print(config)

w2vbert_model2 = create_conformer_shaw_model(config)


print(w2vbert_model2)

cpt = torch.load("conformer_shaw.pt", map_location=torch.device('cpu'))
weights = cpt['model']

w2vbert_model2.load_state_dict(weights)

print("Created W2VBert Large successfully from pretrained model")

block_input = torch.randn(batch_size, seq_len, input_size)
padding_mask = key_padding_mask = torch.zeros(batch_size, seq_len).bool()

w2vbert_model_output2 = w2vbert_model2(block_input, padding_mask)[0]

print(w2vbert_model_output2.sum())
w2vbert_model_output2.sum().backward()

print("Test forward/backward w2vbert model (LARGE) successfully")