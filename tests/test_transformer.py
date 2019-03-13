import argparse

import torch
from torch import nn

import onmt.modules.Transformer
import onmt.modules.Transformer.Layers
import onmt.Constants
import onmt.modules.Loss
from nmtg.data import Dictionary
from nmtg.modules.nmt import NMTModel, NMTEncoder, NMTDecoder
from nmtg.models.transformer.transformer import Transformer
from nmtg.modules.loss import NMTLoss


def make_mask_seq(size, fill):
    maxlen = size[1]
    avg_len = int(fill * maxlen)
    lens = torch.randint(avg_len - 1, avg_len + 2, (size[0],))
    return sequence_mask(lens, maxlen)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


args = argparse.Namespace()
args.layers = 6
args.model_size = 512
args.inner_size = 2048
args.n_heads = 8
args.checkpointing = 0
args.dropout = 0
args.word_dropout = 0
args.attn_dropout = 0
args.emb_dropout = 0
args.residual_dropout = 0
args.weight_norm = False
args.layer_norm = 'fast'
args.activation_layer = 'linear_relu_linear'
args.time = 'positional_encoding'
args.version = 1.0
args.attention_out = 'default'
args.residual_type = 'regular'
args.init_embedding = 'xavier'
args.attention_out = 'default'
args.residual_type = 'regular'
args.param_init = 0.1
args.tie_weights = True
args.batch_first = False
args.max_position_length = 60
args.share_enc_dec_weights = False
args.mask_layers = False
args.single_head_final_layer = False
args.ignore_context = False
args.no_future_masking = False

print("Making embeddings...")
embedding_src = nn.Embedding(30000, args.model_size, padding_idx=onmt.Constants.PAD)
embedding_tgt = embedding_src

onmt.Constants.init_value = args.param_init
onmt.Constants.weight_norm = args.weight_norm
onmt.Constants.attention_out = args.attention_out
onmt.Constants.residual_type = args.residual_type
onmt.Constants.activation_layer = args.activation_layer

print("Making Quan Transformer")
positional_encoder = onmt.modules.Transformer.Layers.PositionalEncoding(args.model_size, len_max=60)

encoder = onmt.modules.Transformer.TransformerEncoder(args, embedding_src, positional_encoder)
decoder = onmt.modules.Transformer.TransformerDecoder(args, embedding_tgt, positional_encoder)

generator = onmt.modules.BaseModel.Generator(args.model_size, 30000)

quan_transformer = onmt.modules.Transformer.Transformer(encoder, decoder, generator)

if args.tie_weights:
    print("* Joining the weights of decoder input and output embeddings")
    quan_transformer.tie_weights()

init = torch.nn.init

init.xavier_uniform_(quan_transformer.generator.linear.weight)

if args.init_embedding == 'xavier':
    init.xavier_uniform_(quan_transformer.encoder.word_lut.weight)
    init.xavier_uniform_(quan_transformer.decoder.word_lut.weight)
elif args.init_embedding == 'normal':
    if quan_transformer.encoder is not None:
        init.normal_(quan_transformer.encoder.word_lut.weight, mean=0, std=args.model_size ** -0.5)
    init.normal_(quan_transformer.decoder.word_lut.weight, mean=0, std=args.model_size ** -0.5)

loss_function_quan = onmt.modules.Loss.NMTLossFunc(30000, label_smoothing=0)

encoder_input = torch.randint(6, 30000, (55, 20)).cuda()
decoder_input = torch.randint(6, 30000, (60, 20)).cuda()
encoder_mask = make_mask_seq((20, 55), 54/55).eq(0).cuda()
decoder_mask = make_mask_seq((20, 60), 54/55).eq(0).cuda()
encoder_input.masked_fill_(encoder_mask.transpose(0, 1), onmt.Constants.PAD)
decoder_input.masked_fill_(decoder_mask.transpose(0, 1), onmt.Constants.PAD)

loss_function_quan.cuda()
quan_transformer.cuda()
optim = torch.optim.Adam(quan_transformer.parameters())


# quan_transformer.encoder.layer_modules[0].multihead.attn_dropout.register_forward_hook(lambda m, i, o: print(i, o))
# quan_transformer.decoder.layer_modules[0].multihead_tgt.register_forward_hook(lambda m, i, o: print(o[0]))

inputs = {'source': encoder_input, 'target_input': decoder_input}
output_dict = quan_transformer(inputs)
outputs_quan = generator(output_dict["hiddens"], False).clone().detach().cpu()
loss_quan = loss_function_quan(output_dict, decoder_input, generator, backward=True)['loss'].clone().detach().cpu()
grads_quan = encoder.layer_modules[0].multihead.fc_query.function.linear.weight.grad.clone().detach().cpu()
grads_quan2 = decoder.layer_modules[-1].multihead_src.fc_concat.function.linear.weight.grad.clone().detach().cpu()
optim.zero_grad()

print("Making Felix Transformer")

dictionary = Dictionary()
dictionary.pad_index = onmt.Constants.PAD
felix_transformer = Transformer.build_model(args)
felix_transformer = NMTModel(NMTEncoder(felix_transformer.encoder, embedding_src, args.word_dropout),
                             NMTDecoder(felix_transformer.decoder, embedding_tgt, args.word_dropout, generator.linear),
                             dictionary, dictionary)
loss_function_felix = NMTLoss(30000, onmt.Constants.PAD, 0.0)
felix_transformer.cuda()
loss_function_felix.cuda()

print(len(list(felix_transformer.parameters())), len(list(quan_transformer.parameters())))
print(sum(p.numel() for p in felix_transformer.parameters()))
print(sum(p.numel() for p in quan_transformer.parameters()))

# share params...
felix_transformer.encoder.encoder.postprocess.layer_norm.function.weight = quan_transformer.encoder.postprocess_layer.layer_norm.function.weight
felix_transformer.encoder.encoder.postprocess.layer_norm.function.bias = quan_transformer.encoder.postprocess_layer.layer_norm.function.bias
for felix_layer, quan_layer in zip(felix_transformer.encoder.encoder.layers, quan_transformer.encoder.layer_modules):
    felix_layer.preprocess_attn.layer_norm.function.weight = quan_layer.preprocess_attn.layer_norm.function.weight
    felix_layer.preprocess_attn.layer_norm.function.bias = quan_layer.preprocess_attn.layer_norm.function.bias
    felix_layer.preprocess_ffn.layer_norm.function.weight = quan_layer.preprocess_ffn.layer_norm.function.weight
    felix_layer.preprocess_ffn.layer_norm.function.bias = quan_layer.preprocess_ffn.layer_norm.function.bias
    felix_layer.feed_forward.function.layer_1.weight = quan_layer.feedforward.function.fc_1.linear.weight
    felix_layer.feed_forward.function.layer_1.bias = quan_layer.feedforward.function.fc_1.linear.bias
    felix_layer.feed_forward.function.layer_2.weight = quan_layer.feedforward.function.fc_2.linear.weight
    felix_layer.feed_forward.function.layer_2.bias = quan_layer.feedforward.function.fc_2.linear.bias
    felix_layer.attention.query_projection.function.weight = quan_layer.multihead.fc_query.function.linear.weight
    felix_layer.attention.query_projection.function.bias = quan_layer.multihead.fc_query.function.linear.bias
    felix_layer.attention.key_projection.function.weight = quan_layer.multihead.fc_key.function.linear.weight
    felix_layer.attention.key_projection.function.bias = quan_layer.multihead.fc_key.function.linear.bias
    felix_layer.attention.value_projection.function.weight = quan_layer.multihead.fc_value.function.linear.weight
    felix_layer.attention.value_projection.function.bias = quan_layer.multihead.fc_value.function.linear.bias
    felix_layer.attention.out_projection.function.weight = quan_layer.multihead.fc_concat.function.linear.weight
    felix_layer.attention.out_projection.function.bias = quan_layer.multihead.fc_concat.function.linear.bias

felix_transformer.decoder.decoder.postprocess.layer_norm.function.weight = quan_transformer.decoder.postprocess_layer.layer_norm.function.weight
felix_transformer.decoder.decoder.postprocess.layer_norm.function.bias = quan_transformer.decoder.postprocess_layer.layer_norm.function.bias
for felix_layer, quan_layer in zip(felix_transformer.decoder.decoder.layers, quan_transformer.decoder.layer_modules):
    felix_layer.preprocess_self_attn.layer_norm.function.weight = quan_layer.preprocess_attn.layer_norm.function.weight
    felix_layer.preprocess_self_attn.layer_norm.function.bias = quan_layer.preprocess_attn.layer_norm.function.bias
    felix_layer.preprocess_ffn.layer_norm.function.weight = quan_layer.preprocess_ffn.layer_norm.function.weight
    felix_layer.preprocess_ffn.layer_norm.function.bias = quan_layer.preprocess_ffn.layer_norm.function.bias
    felix_layer.preprocess_enc_attn.layer_norm.function.weight = quan_layer.preprocess_src_attn.layer_norm.function.weight
    felix_layer.preprocess_enc_attn.layer_norm.function.bias = quan_layer.preprocess_src_attn.layer_norm.function.bias
    felix_layer.feed_forward.function.layer_1.weight = quan_layer.feedforward.function.fc_1.linear.weight
    felix_layer.feed_forward.function.layer_1.bias = quan_layer.feedforward.function.fc_1.linear.bias
    felix_layer.feed_forward.function.layer_2.weight = quan_layer.feedforward.function.fc_2.linear.weight
    felix_layer.feed_forward.function.layer_2.bias = quan_layer.feedforward.function.fc_2.linear.bias
    for felix_attn, quan_attn in zip([felix_layer.self_attention, felix_layer.enc_attention], [quan_layer.multihead_tgt, quan_layer.multihead_src]):
        felix_attn.query_projection.function.weight = quan_attn.fc_query.function.linear.weight
        felix_attn.query_projection.function.bias = quan_attn.fc_query.function.linear.bias
        felix_attn.key_projection.function.weight = quan_attn.fc_key.function.linear.weight
        felix_attn.key_projection.function.bias = quan_attn.fc_key.function.linear.bias
        felix_attn.value_projection.function.weight = quan_attn.fc_value.function.linear.weight
        felix_attn.value_projection.function.bias = quan_attn.fc_value.function.linear.bias
        felix_attn.out_projection.function.weight = quan_attn.fc_concat.function.linear.weight
        felix_attn.out_projection.function.bias = quan_attn.fc_concat.function.linear.bias

# felix_transformer.encoder.encoder.layers[0].attention.attn_dropout.register_forward_hook(lambda m, i, o: print(i, o))
# qfelix_transformer.decoder.decoder.layers[0].self_attention.register_forward_hook(lambda m, i, o: print(o[0]))

outputs, attention_weights = felix_transformer(encoder_input, decoder_input)
outputs_felix = outputs.clone().detach().cpu()
lprobs = felix_transformer.get_normalized_probs(outputs, attention_weights, log_probs=True)
loss = loss_function_felix(lprobs, decoder_input)[0]
loss.backward()
loss_felix = loss.clone().detach().cpu()
grads_felix = felix_transformer.encoder.encoder.layers[0].attention.query_projection.function.weight.grad.clone().detach().cpu()
grads_felix2 = felix_transformer.decoder.decoder.layers[-1].enc_attention.out_projection.function.weight.grad.clone().detach().cpu()

if torch.allclose(outputs_felix, outputs_quan):
    print("Outputs match")
else:
    print("Outputs mismatch:")
    print(outputs_felix)
    print(outputs_quan)
if torch.allclose(grads_felix, grads_quan):
    print("Gradients match")
else:
    print("Gradients mismatch:")
    print(grads_felix)
    print(grads_quan)
    print(torch.max(torch.abs(grads_felix - grads_quan)))
if torch.allclose(grads_felix2, grads_quan2):
    print("Gradients match")
else:
    print("Gradients mismatch:")
    print(grads_felix2)
    print(grads_quan2)
    print(torch.max(torch.abs(grads_felix2 - grads_quan2)))
if torch.allclose(loss_felix, loss_quan):
    print("Losses match")
else:
    print("Loss mismatch:")
    print(loss_felix)
    print(loss_quan)

exit(0)

# test time
felix_command = 'outputs = felix_transformer(encoder_input, decoder_input); ' \
                'lprobs = felix_transformer.get_normalized_probs(outputs, True); ' \
                'loss_function_felix(lprobs, decoder_input); ' \
                'torch.cuda.synchronize()'
quan_command = 'output_dict = quan_transformer(inputs); ' \
               'loss_function_quan(output_dict, decoder_input, generator, backward=False); ' \
               'torch.cuda.synchronize()'
torch.cuda.synchronize()

# Then test speed
import timeit
time = min(timeit.Timer(quan_command, globals=globals()).repeat(10, 100))
print("Quan: {:.3f}ms".format(time * 1000 / 100))
time = min(timeit.Timer(felix_command, globals=globals()).repeat(10, 100))
print("Felix: {:.3f}ms".format(time * 1000 / 100))

