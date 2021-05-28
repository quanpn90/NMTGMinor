import torch
import torch.nn as nn
import onmt
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, Transformer, MixedEncoder
from onmt.models.relative_transformer import RelativeTransformerEncoder, RelativeTransformerDecoder, \
    RelativeTransformer
from onmt.models.transformer_layers import PositionalEncoding
from onmt.models.relative_transformer import SinusoidalPositionalEmbedding, RelativeTransformer
from onmt.modules.copy_generator import CopyGenerator
from onmt.options import backward_compatible
from onmt.constants import add_tokenidx

import math

init = torch.nn.init

MAX_LEN = onmt.constants.max_position_length  # This should be the longest sentence from the dataset


def build_model(opt, dicts):
    # adding missing options if the opt was built before. (for loading old models)
    opt = backward_compatible(opt)

    onmt.constants.layer_norm = opt.layer_norm
    onmt.constants.weight_norm = opt.weight_norm
    onmt.constants.activation_layer = opt.activation_layer
    onmt.constants.version = 1.0
    onmt.constants.attention_out = opt.attention_out
    onmt.constants.residual_type = opt.residual_type
    onmt.constants.fused_ffn = opt.fused_ffn
    opt.nce = opt.nce_noise > 0

    if 'langs' not in dicts:
        dicts['langs'] = {'src': 0, 'tgt': 1}
    opt.n_languages = len(dicts['langs'])

    if opt.bayes_by_backprop:
        from onmt.bayesian_factory import build_model as build_bayesian_model
        model = build_bayesian_model(opt, dicts)
        return model

    if not opt.fusion:
        model = build_tm_model(opt, dicts)
    else:
        raise NotImplementedError
        model = build_fusion(opt, dicts)

    return model


def build_tm_model(opt, dicts):
    onmt.constants = add_tokenidx(opt, onmt.constants, dicts)

    # BUILD POSITIONAL ENCODING
    if opt.time == 'positional_encoding':
        positional_encoder = PositionalEncoding(opt.model_size, len_max=MAX_LEN)
    else:
        raise NotImplementedError

    if opt.reconstruct:
        # reconstruction is only compatible
        assert opt.model == 'relative_transformer'
        assert opt.encoder_type == 'text'

    # BUILD GENERATOR
    if opt.copy_generator:
        if opt.nce_noise > 0:
            print("[INFO] Copy generator overrides NCE.")
            opt.nce = False
            opt.nce_noise = 0
        generators = [CopyGenerator(opt.model_size, dicts['tgt'].size(),
                                    fix_norm=opt.fix_norm_output_embedding)]
    elif opt.nce_noise > 0:
        from onmt.modules.nce.nce_linear import NCELinear
        from onmt.modules.nce.nce_utils import build_unigram_noise
        noise_distribution = build_unigram_noise(torch.FloatTensor(list(dicts['tgt'].frequencies.values())))

        generator = NCELinear(opt.model_size, dicts['tgt'].size(), fix_norm=opt.fix_norm_output_embedding,
                              noise_distribution=noise_distribution, noise_ratio=opt.nce_noise)
        generators = [generator]
    else:
        generators = [onmt.modules.base_seq2seq.Generator(opt.model_size, dicts['tgt'].size(),
                                                          fix_norm=opt.fix_norm_output_embedding)]

    # BUILD EMBEDDINGS
    if 'src' in dicts:
        if (not hasattr(opt, "enc_pretrained_model")) or (not opt.enc_pretrained_model):
            embedding_src = nn.Embedding(dicts['src'].size(),
                                         opt.model_size,
                                         padding_idx=onmt.constants.SRC_PAD)
    else:
        embedding_src = None

    if opt.join_embedding and embedding_src is not None:
        embedding_tgt = embedding_src
        print("* Joining the weights of encoder and decoder word embeddings")
    elif not opt.dec_pretrained_model:
        embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                     opt.model_size,
                                     padding_idx=onmt.constants.TGT_PAD)
    else:
        assert opt.model == "pretrain_transformer"
        embedding_tgt = None

    if opt.use_language_embedding:
        print("* Create language embeddings with %d languages" % len(dicts['langs']))
        language_embeddings = nn.Embedding(len(dicts['langs']), opt.model_size)
    else:
        language_embeddings = None

    if opt.ctc_loss != 0:
        generators.append(onmt.modules.base_seq2seq.Generator(opt.model_size, dicts['tgt'].size() + 1))

    if opt.model in ['conformer', 'speech_transformer', 'hybrid_transformer']:
        onmt.constants.init_value = opt.param_init
        from onmt.models.speech_recognizer.relative_transformer import \
            SpeechTransformerEncoder, SpeechTransformerDecoder

        if opt.model == 'conformer':
            from onmt.models.speech_recognizer.conformer import ConformerEncoder, Conformer
            from onmt.models.speech_recognizer.lstm import SpeechLSTMDecoder
            opt.cnn_downsampling = True  # force this bool to have masking at decoder to be corrected
            encoder = ConformerEncoder(opt, None, None, 'audio')

            # decoder = SpeechLSTMDecoder(opt, embedding_tgt, language_embeddings=language_embeddings)
            decoder = SpeechTransformerDecoder(opt, embedding_tgt, positional_encoder,
                                               language_embeddings=language_embeddings)

            # model = Conformer(encoder, decoder, nn.ModuleList(generators), ctc=opt.ctc_loss > 0.0)
            model = RelativeTransformer(encoder, decoder, nn.ModuleList(generators),
                                        None, None, mirror=opt.mirror_loss, ctc=opt.ctc_loss > 0.0)
        elif opt.model == 'hybrid_transformer':
            from onmt.models.speech_recognizer.lstm import SpeechLSTMDecoder, SpeechLSTMEncoder, SpeechLSTMSeq2Seq
            encoder = SpeechTransformerEncoder(opt, None, positional_encoder, opt.encoder_type)

            decoder = SpeechLSTMDecoder(opt, embedding_tgt, language_embeddings=language_embeddings)

            model = SpeechLSTMSeq2Seq(encoder, decoder, nn.ModuleList(generators), ctc=opt.ctc_loss > 0.0)
        else:
            encoder = SpeechTransformerEncoder(opt, None, positional_encoder, opt.encoder_type)

            decoder = SpeechTransformerDecoder(opt, embedding_tgt, positional_encoder,
                                               language_embeddings=language_embeddings)
            model = RelativeTransformer(encoder, decoder, nn.ModuleList(generators),
                                        None, None, mirror=opt.mirror_loss, ctc=opt.ctc_loss > 0.0)

        # If we use the multilingual model and weights are partitioned:
        if opt.multilingual_partitioned_weights:
            # this is basically the language embeddings
            factor_embeddings = nn.Embedding(len(dicts['langs']), opt.mpw_factor_size)

            encoder.factor_embeddings = factor_embeddings
            decoder.factor_embeddings = factor_embeddings

    elif opt.model in ["LSTM", 'lstm']:
        # print("LSTM")
        onmt.constants.init_value = opt.param_init
        from onmt.models.speech_recognizer.lstm import SpeechLSTMDecoder, SpeechLSTMEncoder, SpeechLSTMSeq2Seq

        encoder = SpeechLSTMEncoder(opt, None, opt.encoder_type)

        decoder = SpeechLSTMDecoder(opt, embedding_tgt, language_embeddings=language_embeddings)

        model = SpeechLSTMSeq2Seq(encoder, decoder, nn.ModuleList(generators), ctc=opt.ctc_loss > 0.0)

    elif opt.model in ['multilingual_translator', 'translator']:
        onmt.constants.init_value = opt.param_init
        from onmt.models.multilingual_translator.relative_transformer import \
            RelativeTransformerEncoder, RelativeTransformerDecoder

        encoder = RelativeTransformerEncoder(opt, embedding_src, None,
                                             opt.encoder_type, language_embeddings=language_embeddings)
        decoder = RelativeTransformerDecoder(opt, embedding_tgt, None, language_embeddings=language_embeddings)

        model = RelativeTransformer(encoder, decoder, nn.ModuleList(generators),
                                    None, None, mirror=opt.mirror_loss)

    elif opt.model in ['transformer', 'stochastic_transformer']:
        onmt.constants.init_value = opt.param_init

        if opt.encoder_type == "text":
            encoder = TransformerEncoder(opt, embedding_src, positional_encoder,
                                         opt.encoder_type, language_embeddings=language_embeddings)
        elif opt.encoder_type == "audio":
            encoder = TransformerEncoder(opt, None, positional_encoder, opt.encoder_type)
        elif opt.encoder_type == "mix":
            text_encoder = TransformerEncoder(opt, embedding_src, positional_encoder,
                                              "text", language_embeddings=language_embeddings)
            audio_encoder = TransformerEncoder(opt, None, positional_encoder, "audio")
            encoder = MixedEncoder(text_encoder, audio_encoder)
        else:
            print("Unknown encoder type:", opt.encoder_type)
            exit(-1)

        decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder, language_embeddings=language_embeddings)

        model = Transformer(encoder, decoder, nn.ModuleList(generators), mirror=opt.mirror_loss)

    elif opt.model == 'relative_transformer':
        from onmt.models.relative_transformer import \
            RelativeTransformerEncoder, RelativeTransformerDecoder

        if opt.encoder_type == "text":
            encoder = RelativeTransformerEncoder(opt, embedding_src, None,
                                                 opt.encoder_type, language_embeddings=language_embeddings)
        if opt.encoder_type == "audio":
            # raise NotImplementedError
            encoder = RelativeTransformerEncoder(opt, None, None, encoder_type=opt.encoder_type,
                                                 language_embeddings=language_embeddings)

        generator = nn.ModuleList(generators)
        decoder = RelativeTransformerDecoder(opt, embedding_tgt, None, language_embeddings=language_embeddings)

        if opt.reconstruct:
            rev_decoder = RelativeTransformerDecoder(opt, embedding_src, None, language_embeddings=language_embeddings)
            rev_generator = [onmt.modules.base_seq2seq.Generator(opt.model_size, dicts['src'].size(),
                                                                 fix_norm=opt.fix_norm_output_embedding)]
            rev_generator = nn.ModuleList(rev_generator)
        else:
            rev_decoder = None
            rev_generator = None

        model = RelativeTransformer(encoder, decoder, generator, rev_decoder, rev_generator, mirror=opt.mirror_loss)

    elif opt.model == 'universal_transformer':
        from onmt.legacy.old_models.universal_transformer import UniversalTransformerDecoder, UniversalTransformerEncoder

        generator = nn.ModuleList(generators)

        if opt.encoder_type == "text":
            encoder = UniversalTransformerEncoder(opt, embedding_src, positional_encoder,
                                                  opt.encoder_type, language_embeddings=language_embeddings)
        elif opt.encoder_type == "audio":
            encoder = UniversalTransformerEncoder(opt, None, positional_encoder, opt.encoder_type)

        decoder = UniversalTransformerDecoder(opt, embedding_tgt, positional_encoder,
                                              language_embeddings=language_embeddings)

        model = Transformer(encoder, decoder, generator, mirror=opt.mirror_loss)

    elif opt.model == 'pretrain_transformer':
        assert (opt.enc_pretrained_model or opt.dec_pretrained_model)
        from onmt.models.pretrain_transformer import PretrainTransformer
        print(f"pos_emb_type: {opt.pos_emb_type}")
        print(f"max_pos_length: {opt.max_pos_length }")
        print(f"Share position embeddings cross heads: {not opt.diff_head_pos}")
        print()
        if opt.enc_pretrained_model:
            print("* Build encoder with enc_pretrained_model: {}".format(opt.enc_pretrained_model))
        if opt.enc_pretrained_model == "bert":
            from pretrain_module.configuration_bert import BertConfig
            from pretrain_module.modeling_bert import BertModel

            enc_bert_config = BertConfig.from_json_file(opt.enc_config_file)
            encoder = BertModel(enc_bert_config,
                                bert_word_dropout=opt.enc_pretrain_word_dropout,
                                bert_emb_dropout=opt.enc_pretrain_emb_dropout,
                                bert_atten_dropout=opt.enc_pretrain_attn_dropout,
                                bert_hidden_dropout=opt.enc_pretrain_hidden_dropout,
                                bert_hidden_size=opt.enc_pretrain_hidden_size,
                                is_decoder=False,
                                before_plm_output_ln=opt.before_enc_output_ln,
                                gradient_checkpointing=opt.enc_gradient_checkpointing,
                                max_pos_len=opt.max_pos_length,
                                diff_head_pos=opt.diff_head_pos,
                                pos_emb_type=opt.pos_emb_type,
                                )

        elif opt.enc_pretrained_model == "roberta":
            from pretrain_module.configuration_roberta import RobertaConfig
            from pretrain_module.modeling_roberta import RobertaModel
            enc_roberta_config = RobertaConfig.from_json_file(opt.enc_config_file)

            encoder = RobertaModel(enc_roberta_config,
                                   bert_word_dropout=opt.enc_pretrain_word_dropout,
                                   bert_emb_dropout=opt.enc_pretrain_emb_dropout,
                                   bert_atten_dropout=opt.enc_pretrain_attn_dropout,
                                   bert_hidden_dropout=opt.enc_pretrain_hidden_dropout,
                                   bert_hidden_size=opt.enc_pretrain_hidden_size,
                                   is_decoder=False,
                                   before_plm_output_ln=opt.before_enc_output_ln,
                                   gradient_checkpointing=opt.enc_gradient_checkpointing,
                                   max_pos_len=opt.max_pos_length,
                                   diff_head_pos=opt.diff_head_pos,
                                   pos_emb_type=opt.pos_emb_type,
                                   )
        elif not opt.enc_pretrained_model:
            print(" Encoder is not from pretrained model")
            encoder = TransformerEncoder(opt, embedding_src, positional_encoder,
                                         opt.encoder_type, language_embeddings=language_embeddings)
        else:
            print("Warning: only bert and roberta are implemented for encoder")
            exit(-1)

        if opt.load_from or not opt.enc_state_dict:
            if opt.verbose:
                print("  No weights loading from {} for encoder".format(opt.enc_pretrained_model))
        elif opt.enc_pretrained_model:
            print("  Loading weights for encoder from: \n", opt.enc_state_dict)

            enc_model_state_dict = torch.load(opt.enc_state_dict, map_location="cpu")

            encoder.from_pretrained(state_dict=enc_model_state_dict,
                                    model=encoder,
                                    output_loading_info=opt.verbose,
                                    model_prefix=opt.enc_pretrained_model
                                    )

        if opt.dec_pretrained_model:
            print("* Build decoder with dec_pretrained_model: {}".format(opt.dec_pretrained_model))

        if opt.dec_pretrained_model == "bert":
            if opt.enc_pretrained_model != "bert":
                from pretrain_module.configuration_bert import BertConfig
                from pretrain_module.modeling_bert import BertModel
            dec_bert_config = BertConfig.from_json_file(opt.dec_config_file)
            decoder = BertModel(dec_bert_config,
                                bert_word_dropout=opt.dec_pretrain_word_dropout,
                                bert_emb_dropout=opt.dec_pretrain_emb_dropout,
                                bert_atten_dropout=opt.dec_pretrain_attn_dropout,
                                bert_hidden_dropout=opt.dec_pretrain_hidden_dropout,
                                bert_hidden_size=opt.dec_pretrain_hidden_size,
                                is_decoder=True,
                                gradient_checkpointing=opt.dec_gradient_checkpointing,
                                max_pos_len=opt.max_pos_length,
                                diff_head_pos=opt.diff_head_pos,
                                pos_emb_type=opt.pos_emb_type,
                                )

        elif opt.dec_pretrained_model == "roberta":
            if opt.enc_pretrained_model != "roberta":
                from pretrain_module.configuration_roberta import RobertaConfig
                from pretrain_module.modeling_roberta import RobertaModel

            dec_roberta_config = RobertaConfig.from_json_file(opt.dec_config_file)

            decoder = RobertaModel(dec_roberta_config,
                                   bert_word_dropout=opt.dec_pretrain_word_dropout,
                                   bert_emb_dropout=opt.dec_pretrain_emb_dropout,
                                   bert_atten_dropout=opt.dec_pretrain_attn_dropout,
                                   bert_hidden_dropout=opt.dec_pretrain_hidden_dropout,
                                   bert_hidden_size=opt.dec_pretrain_hidden_size,
                                   is_decoder=True,
                                   gradient_checkpointing=opt.dec_gradient_checkpointing,
                                   max_pos_len=opt.max_pos_length,
                                   diff_head_pos=opt.diff_head_pos,
                                   pos_emb_type=opt.pos_emb_type,
                                   )

        elif not opt.dec_pretrained_model:
            print(" Decoder is not from pretrained model")
            decoder = TransformerDecoder(opt, embedding_tgt, positional_encoder,
                                         language_embeddings=language_embeddings)
        else:
            print("Warning: only bert and roberta are implemented for decoder")
            exit(-1)

        if opt.load_from or not opt.dec_state_dict:
            if opt.verbose:
                print("  No weights loading from {} for decoder".format(opt.dec_pretrained_model))
        elif opt.enc_pretrained_model:
            print("  Loading weights for decoder from: \n", opt.dec_state_dict)
            dec_model_state_dict = torch.load(opt.dec_state_dict, map_location="cpu")

            decoder.from_pretrained(state_dict=dec_model_state_dict,
                                    model=decoder,
                                    output_loading_info=opt.verbose,
                                    model_prefix=opt.dec_pretrained_model
                                    )

        encoder.enc_pretrained_model = opt.enc_pretrained_model
        decoder.dec_pretrained_model = opt.dec_pretrained_model

        encoder.input_type = opt.encoder_type

        model = PretrainTransformer(encoder, decoder, nn.ModuleList(generators))
    else:
        raise NotImplementedError

    if opt.tie_weights:
        print("* Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    return model


def init_model_parameters(model, opt):
    """
    Initializing model parameters. Mostly using normal distribution (0, std)
    """
    init_std = 0.02  # magical number

    # opt.init something ...

    def init_weight(weight):
        if opt.init == 'normal':
            if len(weight.shape) == 2:
                std_ = math.sqrt(2.0 / (weight.shape[0] + weight.shape[1]))
                nn.init.normal_(weight, 0.0, std_)
            else:
                nn.init.normal_(weight, 0.0, init_std)
        elif opt.init == 'uniform':
            if len(weight.shape) == 2:
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.uniform_(weight, -init_std, init_std)

    def init_embed(weight, padding_idx=0):

        # The embedding is intialized as in "Attention is all you need" and "Ada-factor" paper
        std_ = opt.model_size ** -0.5 if not opt.rezero else 0.05

        if opt.init_embedding == 'normal':
            nn.init.normal_(weight, 0.0, std_)
        if opt.init_embedding == 'fixed':
            nn.init.normal_(weight, 0.0, 0.01)
        else:  # uni form
            nn.init.uniform_(weight, -std_, std_)

        # don't uncomment the next lines...
        # for some reason normalizing the weights at fp16 doesnt work when setting the padding to 0
        # if not opt.fix_norm_output_embedding:
        #     nn.init.constant_(weight[padding_idx], 0)

    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('Embedding') != -1:

            initialize = True
            if hasattr(m, "no_need_to_initialize"):
                if m.no_need_to_initialize:
                    initialize = False
            if initialize:
                if hasattr(m, 'weight') and hasattr(m, 'padding_idx'):
                    init_embed(m.weight, m.padding_idx)
            # nn.init.constant_(m.weight[m.padding_idx], 0.0)

        elif classname.find('LayerNorm') != -1 or classname.find('FusedLayerNorm') != -1:
            if hasattr(m, 'weight'):
                # if opt.init == 'normal':
                #     nn.init.normal_(m.weight, 1.0, 0)
                # else:
                #     nn.init.uniform_(m.weight, 1.0 - init_std, 1.0 + init_std)
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
            pass
        elif classname.find('RelativeTransformerEncoder') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)
        elif classname.find('RelativeTransformerDecoder') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)
        elif classname.find('RelPartialLearnableMultiHeadAttn') != -1:
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
        elif classname.find('EncdecMultiheadAttn') != -1:
            m.reset_parameters(init=opt.init)
        elif classname.find('RelativeSelfMultiheadAttn') != -1:
            m.reset_parameters(init=opt.init)
        elif classname.find('PositionWiseFeedForward') != -1:
            m.reset_parameters(init=opt.init)

    if opt.model != "pretrain_transformer":
        print('Initializing entire model parameters')
        model.apply(weights_init)
    else:
        if opt.enc_pretrained_model and not opt.dec_pretrained_model:
            print('Initializing only decoder parameters')
            model.decoder.apply(weights_init)
        if not opt.enc_pretrained_model and opt.dec_pretrained_model:
            print('Initializing only encoder parameters')
            model.encoder.apply(weights_init)

    if hasattr(model, 'decoder'):
        if not opt.dec_pretrained_model:
            model.decoder.word_lut.apply(weights_init)
    else:
        model.tgt_embedding.apply(weights_init)

    if opt.multilingual_partitioned_weights:
        factor_embeddings = model.encoder.factor_embeddings

        # this embedding scheme avoids a large initial perplexity
        # basically an on-off switch to start with
        with torch.no_grad():
            # factor_embeddings.weight.bernoulli_(0.5).mul_(-2).add_(1)
            factor_embeddings.weight.uniform_(-1, 1)
    return


def build_language_model(opt, dicts):
    opt = backward_compatible(opt)

    onmt.constants.layer_norm = opt.layer_norm
    onmt.constants.weight_norm = opt.weight_norm
    onmt.constants.activation_layer = opt.activation_layer
    onmt.constants.version = 1.0
    onmt.constants.attention_out = opt.attention_out
    onmt.constants.residual_type = opt.residual_type

    from onmt.models.transformer_xl import TransformerXL

    embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                 opt.model_size,
                                 padding_idx=onmt.constants.TGT_PAD)

    if opt.use_language_embedding:
        print("* Create language embeddings with %d languages" % len(dicts['langs']))
        language_embeddings = nn.Embedding(len(dicts['langs']), opt.model_size)
    else:
        language_embeddings = None

    generators = [onmt.modules.base_seq2seq.Generator(opt.model_size, dicts['tgt'].size())]

    model = TransformerXL(opt, embedding_tgt, nn.ModuleList(generators), language_embeddings=language_embeddings)

    model.tgt_dict = dicts['tgt']

    if opt.tie_weights:
        print("* Joining the weights of decoder input and output embeddings")
        model.tie_weights()

    return model


def build_fusion(opt, dicts):
    # the fusion model requires a pretrained language model
    print("Loading pre-trained language model from %s" % opt.lm_checkpoint)
    lm_checkpoint = torch.load(opt.lm_checkpoint, map_location=lambda storage, loc: storage)

    # first we build the lm model and lm checkpoint
    lm_opt = lm_checkpoint['opt']

    lm_model = build_language_model(lm_opt, dicts)

    # load parameter for pretrained model
    lm_model.load_state_dict(lm_checkpoint['model'])

    # main model for seq2seq (translation, asr)
    tm_model = build_tm_model(opt, dicts)

    from onmt.legacy.FusionNetwork.Models import FusionNetwork
    model = FusionNetwork(tm_model, lm_model)

    return model


def optimize_model(model, fp16=True, distributed=False):
    """
    Used to potentially upgrade the components with more optimized counterparts in the future
    """

    def replace_layer_norm(m, name):

        replacable = True
        try:
            # from apex.normalization.fused_layer_norm import FusedLayerNorm
            import importlib
            from apex.normalization.fused_layer_norm import FusedLayerNorm
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        except ModuleNotFoundError:
            replacable = False

        if replacable:
            for attr_str in dir(m):
                target_attr = getattr(m, attr_str)
                if type(target_attr) == torch.nn.LayerNorm:
                    setattr(m, attr_str, FusedLayerNorm(target_attr.normalized_shape,
                                                        eps=target_attr.eps,
                                                        elementwise_affine=target_attr.elementwise_affine))
            for n, ch in m.named_children():
                replace_layer_norm(ch, n)

    def safe_batch_norm(m, name):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if type(target_attr) == torch.nn.BatchNorm2d or type(target_attr) == torch.nn.BatchNorm1d:

                if fp16:
                    target_attr.eps = 1e-5  # tiny value for fp16 according to AllenNLP

                setattr(m, attr_str, target_attr)

    # replace_layer_norm(model, "Transformer")


def freeze_model_specialized_weights(model):
    from onmt.modules.multilingual_factorized.linear import MFWPositionWiseFeedForward
    from onmt.modules.multilingual_factorized.encdec_attention import MFWEncdecMultiheadAttn
    from onmt.modules.multilingual_factorized.relative_attention import MFWRelativeSelfMultiheadAttn

    def freeze(m):
        classname = m.__class__.__name__

        if classname in ['MFWPositionWiseFeedForward',
                         "MFWEncdecMultiheadAttn",
                         "MFWRelativeSelfMultiheadAttn"]:
            m.freeze()

    model.apply(freeze)

    return


def unfreeze_model_speciailized_weights(model):

    def unfreeze(m):
        classname = m.__class__.__name__

        if classname in ['MFWPositionWiseFeedForward',
                         "MFWEncdecMultiheadAttn",
                         "MFWRelativeSelfMultiheadAttn"]:
            m.unfreeze()

    model.apply(unfreeze)

    return
