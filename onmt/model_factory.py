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
import json
from types import SimpleNamespace

init = torch.nn.init

MAX_LEN = onmt.constants.max_position_length  # This should be the longest sentence from the dataset


def json_to_namespace(json_file):
    with open(json_file) as f:
        x = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    for name in x.__dict__:
        if x.__dict__[name] in ['False', 'True']:
            x.__dict__[name] = (x.__dict__[name] == 'True')
    return x


def remove_pretrain_weights(opt):
    opt.dec_state_dict = ""
    opt.enc_state_dict = ""
    return opt


def build_model(opt, dicts, remove_pretrain=False, constants=None):
    # adding missing options if the opt was built before. (for loading old models)
    opt = backward_compatible(opt)
    if remove_pretrain:
        print("[INFO] Removing pretrained weights from opt")
        opt = remove_pretrain_weights(opt)

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
    opt.n_attributes = len(dicts['atbs']) if 'atbs' in dicts else 0

    if 'atbs' in dicts and 'nothingness' in dicts['atbs'] and len(dicts['atbs']) == 1:
        opt.n_attributes = 0

    if opt.bayes_by_backprop:
        from onmt.bayesian_factory import build_model as build_bayesian_model
        model = build_bayesian_model(opt, dicts)
        return model

    if not opt.fusion:
        model = build_tm_model(opt, dicts, constants=constants)
    else:
        raise NotImplementedError
        model = build_fusion(opt, dicts)

    return model


def build_classifier(opt, dicts):
    opt = backward_compatible(opt)
    if 'langs' not in dicts:
        dicts['langs'] = {'src': 0, 'tgt': 1}
    opt.n_languages = len(dicts['langs'])

    generators = [onmt.modules.base_seq2seq.Generator(opt.model_size, dicts['tgt'].size(),
                                                      fix_norm=opt.fix_norm_output_embedding)]

    onmt.constants.init_value = opt.param_init
    from onmt.models.speech_recognizer.relative_transformer import \
        SpeechTransformerEncoder, SpeechTransformerDecoder

    from onmt.models.speech_recognizer.classifier import TransformerClassifier

    if opt.model in ["wav2vec2", "wav2vec"]:
        from onmt.models.speech_recognizer.wav2vec2 import FairseqWav2Vec, Wav2vecBERT

        encoder = FairseqWav2Vec(opt, model_path=opt.wav2vec2_pretrained_model)

    elif opt.model in ["LSTM", 'lstm']:
        # print("LSTM")
        onmt.constants.init_value = opt.param_init
        from onmt.models.speech_recognizer.lstm import SpeechLSTMDecoder, SpeechLSTMEncoder, SpeechLSTMSeq2Seq

        encoder = SpeechLSTMEncoder(opt, None, opt.encoder_type)
    else:
        encoder = SpeechTransformerEncoder(opt, None, None, opt.encoder_type)

    model = TransformerClassifier(encoder, nn.ModuleList(generators), mpc=opt.mpc)

    return model


def build_tm_model(opt, dicts, constants=None):
    # onmt.constants = add_tokenidx(opt, onmt.constants, dicts)
    if constants is None:
        constants = onmt.constants

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
                                         padding_idx=constants.SRC_PAD)
    else:
        embedding_src = None

    if opt.join_embedding and embedding_src is not None:
        embedding_tgt = embedding_src
        print("* Joining the weights of encoder and decoder word embeddings")
    elif not opt.dec_pretrained_model:
        embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                     opt.model_size,
                                     padding_idx=constants.TGT_PAD)
    else:
        assert opt.model in ["pretrain_transformer", "wav2vec2_bert",
                             "wav2vec_mbart50", "quantize_wav2vec2_bert", "quantize_wav2vec2_mbart50"], \
            "Expecting a pretrained model that has a " \
            "separate Embedding initialization"
        embedding_tgt = None

    if opt.use_language_embedding:
        print("* Create language embeddings with %d languages" % len(dicts['langs']))
        language_embeddings = nn.Embedding(len(dicts['langs']), opt.model_size)
    else:
        language_embeddings = None

    if opt.model in ['wav2vec2_bert', 'quantize_wav2vec2_bert', 'quantize_wav2vec2_mbart50']:
        from onmt.models.speech_recognizer.wav2vec2 import FairseqWav2Vec, Wav2vecBERT

        # if opt.model.startswith("quantize"):
        #     from pretrain_module.modeling_mbart import MBartDecoder, MBartEncoder
        #     from pretrain_module.configuration_mbart import MBartConfig
        #     enc_mbart_config = MBartConfig.from_json_file(opt.enc_config_file)
        #     discrete_encoder = MBartEncoder(enc_mbart_config, opt)
        #     # print("[INFO] Loading weights for mBART encoder from: %s ..." % opt.enc_state_dict)
        #     # enc_model_state_dict = torch.load(opt.enc_state_dict, map_location="cpu")
        #     # discrete_encoder.load_state_dict(enc_model_state_dict)
        # else:
        #     discrete_encoder = None

        # TODO: create a stacked encoder here
        # if len(opt.dec_pretrained_model)
        stacked_encoder = None
        if len(opt.enc_stacked_pretrained_model) > 0:
            if "mbart" in opt.enc_stacked_pretrained_model:
                print("[INFO] Created a stacked encoder MBART-50")
                from pretrain_module.modeling_mbart import MBartEncoder
                from pretrain_module.configuration_mbart import MBartConfig
                enc_mbart_config = MBartConfig.from_json_file(opt.enc_config_file)
                stacked_encoder = MBartEncoder(enc_mbart_config, opt)
            else:
                raise NotImplementedError

            if opt.enc_state_dict is not None and len(opt.enc_state_dict) > 1:
                print("[INFO] Loading weights for stacked encoder from: %s ..." % opt.enc_state_dict)
                enc_model_state_dict = torch.load(opt.enc_state_dict, map_location="cpu")

                # load parameters from state dict to model (using huggingface's approach)
                # decoder.from_pretrained(state_dict=dec_model_state_dict,
                #                         model=decoder,
                #                         output_loading_info=opt.verbose,
                #                         model_prefix=opt.dec_pretrained_model
                #                         )
                # current_dict = decoder.state_dict()
                #
                # for key in current_dict:
                #     if key not in dec_model_state_dict:
                #         dec_model_state_dict[key] = current_dict[key]

                stacked_encoder.load_state_dict(enc_model_state_dict)
                print("[INFO] ... Done")

        discrete_encoder = None
        encoder = FairseqWav2Vec(opt, model_path=opt.wav2vec2_pretrained_model,
                                 discrete_encoder=discrete_encoder, stacked_encoder=stacked_encoder)

        sub_encoder = None

        if "mbart" in opt.dec_pretrained_model:
            from pretrain_module.configuration_mbart import MBartConfig
            from pretrain_module.modeling_mbart import MBartDecoder, MBartEncoder
            print("[INFO] Created MBART decoder from: %s ..." % opt.dec_config_file)
            dec_mbart_config = MBartConfig.from_json_file(opt.dec_config_file)

            decoder = MBartDecoder(dec_mbart_config, opt)

            if opt.freeze_embedding:
                decoder.embed_tokens.weight.requires_grad = False

            # if opt.enc_config_file:
            #     enc_mbart_config = MBartConfig.from_json_file(opt.enc_config_file)
            #     sub_encoder = MBartEncoder(enc_mbart_config, opt)
        elif opt.dec_pretrained_model in ['deltalm']:
            print("[INFO] Created DeltaLM decoder from: %s ..." % opt.dec_config_file)
            from onmt.models.deltalm.deltalm import DeltaLMDecoder
            deltalm_config = json_to_namespace(opt.dec_config_file)
            print(constants)
            embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                         deltalm_config.decoder_embed_dim,
                                         padding_idx=constants.TGT_PAD)
            decoder = DeltaLMDecoder(deltalm_config, embedding_tgt)
            # from pretrain_module.configuration_deltalm import DeltaMConfig
            # from pretrain_module.modeling_deltalm import DeltaLMDecoder
            # print("[INFO] Created DeltaLM decoder from: %s ..." % opt.dec_config_file)
            # dec_mbart_config = DeltaLMConfig.from_json_file(opt.dec_config_file)
            # decoder = DeltaLMDecoder(dec_mbart_config, opt)

            generators[0].linear.weight = decoder.embed_tokens.weight
            if opt.freeze_embedding:
                decoder.embed_tokens.weight.requires_grad = False

        elif opt.dec_pretrained_model == "bart":
            from pretrain_module.configuration_bart import BartConfig
            from pretrain_module.modeling_bart import BartDecoder

            dec_bart_config = BartConfig.from_json_file(opt.dec_config_file)

            decoder = BartDecoder(dec_bart_config, opt)

        if opt.dec_state_dict is not None and len(opt.dec_state_dict) > 1:
            print("[INFO] Loading weights for decoder from: %s ..." % opt.dec_state_dict)
            dec_model_state_dict = torch.load(opt.dec_state_dict, map_location="cpu")

            # load parameters from state dict to model (using huggingface's approach)
            # decoder.from_pretrained(state_dict=dec_model_state_dict,
            #                         model=decoder,
            #                         output_loading_info=opt.verbose,
            #                         model_prefix=opt.dec_pretrained_model
            #                         )
            current_dict = decoder.state_dict()

            for key in current_dict:
                if key not in dec_model_state_dict:
                    dec_model_state_dict[key] = current_dict[key]

            decoder.load_state_dict(dec_model_state_dict)
            print("[INFO] ... Done")

        # if len(opt.enc_state_dict) > 1:
        #     print("[INFO] Loading weights for mBART encoder from: %s ..." % opt.enc_state_dict)
        #     enc_model_state_dict = torch.load(opt.enc_state_dict, map_location="cpu")
        #     sub_encoder.load_state_dict(enc_model_state_dict)
        #     for parameter in sub_encoder.parameters():
        #         parameter.requires_grad = False # don't update these guys
        #         sub_encoder.embed_tokens = decoder.embed_tokens # and reduce memory usage

        decoder.dec_pretrained_model = opt.dec_pretrained_model
        if opt.freeze_embedding:
            generators[0].linear.bias.requires_grad = False

        model = Wav2vecBERT(encoder, decoder, nn.ModuleList(generators), mirror=opt.mirror_loss, ctc=opt.ctc_loss > 0.0,
                            sub_encoder=sub_encoder)

        # TODO: share the ctc_loss weight with the decoder weights

    elif opt.model in ['wav2vec2_transformer']:
        from onmt.models.speech_recognizer.wav2vec2 import FairseqWav2Vec, Wav2vecTransformer
        from onmt.models.speech_recognizer.relative_transformer import SpeechTransformerDecoder

        encoder = FairseqWav2Vec(opt, model_path=opt.wav2vec2_pretrained_model)

        decoder = SpeechTransformerDecoder(opt, embedding_tgt, positional_encoder,
                                           language_embeddings=language_embeddings)

        model = Wav2vecTransformer(encoder, decoder, nn.ModuleList(generators),
                                   mirror=opt.mirror_loss, ctc=opt.ctc_loss > 0.0)

    elif opt.model in ['discourse_speech_transformer']:
        from onmt.models.discourse.discourse_transformer import DiscourseTransformerEncoder, DiscourseTransformer
        from onmt.models.speech_recognizer.relative_transformer import \
            SpeechTransformerEncoder, SpeechTransformerDecoder

        encoder = SpeechTransformerEncoder(opt, None, positional_encoder, opt.encoder_type)

        decoder = SpeechTransformerDecoder(opt, embedding_tgt, positional_encoder,
                                           language_embeddings=language_embeddings)

        encoder = DiscourseTransformerEncoder(opt, encoder=encoder)

        model = DiscourseTransformer(encoder, decoder, nn.ModuleList(generators),
                                     None, None, mirror=opt.mirror_loss, ctc=opt.ctc_loss > 0.0)

    elif opt.model in ['discourse_translator']:
        from onmt.models.discourse.discourse_transformer import DiscourseTransformerEncoder, DiscourseTransformer
        onmt.constants.init_value = opt.param_init
        from onmt.models.multilingual_translator.relative_transformer import \
            RelativeTransformerEncoder, RelativeTransformerDecoder

        encoder = RelativeTransformerEncoder(opt, embedding_src, None,
                                             opt.encoder_type, language_embeddings=language_embeddings)
        decoder = RelativeTransformerDecoder(opt, embedding_tgt, None, language_embeddings=language_embeddings)

        encoder = DiscourseTransformerEncoder(opt, encoder=encoder)

        model = DiscourseTransformer(encoder, decoder, nn.ModuleList(generators),
                                     None, None, mirror=opt.mirror_loss)

    elif opt.model in ['conformer', 'speech_transformer', 'hybrid_transformer']:
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
        from onmt.legacy.old_models.universal_transformer import UniversalTransformerDecoder, \
            UniversalTransformerEncoder

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

        if opt.enc_pretrained_model in ["mbart", "mbart50"]:
            from pretrain_module.configuration_mbart import MBartConfig
            from pretrain_module.modeling_mbart import MBartEncoder
            enc_mbart_config = MBartConfig.from_json_file(opt.enc_config_file)

            encoder = MBartEncoder(enc_mbart_config, opt)

        elif opt.enc_pretrained_model in ["m2m", "m2m100"]:
            from pretrain_module.configuration_m2m100 import M2M100Config
            from pretrain_module.modeling_m2m100 import M2M100Encoder
            enc_mbart_config = M2M100Config.from_json_file(opt.enc_config_file)

            encoder = M2M100Encoder(enc_mbart_config, opt)

        elif opt.enc_pretrained_model in ["deltalm"]:
            from onmt.models.deltalm.deltalm import DeltaLMEncoder

            deltalm_config = json_to_namespace(opt.dec_config_file)
            embedding_src = nn.Embedding(dicts['src'].size(),
                                         deltalm_config.encoder_embed_dim,
                                         padding_idx=constants.SRC_PAD)
            encoder = DeltaLMEncoder(deltalm_config, embedding_src)

        elif not opt.enc_pretrained_model:
            print(" Encoder is not from pretrained model")
            encoder = TransformerEncoder(opt, embedding_src, positional_encoder,
                                         opt.encoder_type, language_embeddings=language_embeddings)
        else:
            print("Pretrained Encoder type not supported")
            exit(-1)

        if opt.load_from or not opt.enc_state_dict:
            if opt.verbose:
                print("  No weights loading from {} for encoder".format(opt.enc_pretrained_model))
        elif opt.enc_pretrained_model:
            print("[INFO] Loading weights for encoder from: \n", opt.enc_state_dict)

            enc_model_state_dict = torch.load(opt.enc_state_dict, map_location="cpu")

            if opt.enc_pretrained_model not in ["deltalm"]:
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
                                max_pos_len=opt.max_pos_length,
                                pos_emb_type=opt.pos_emb_type,
                                )

        elif opt.dec_pretrained_model in ["mbart", "mbart50"]:
            if opt.enc_pretrained_model not in ["mbart", "mbart50"]:
                from pretrain_module.configuration_mbart import MBartConfig
            from pretrain_module.modeling_mbart import MBartDecoder

            dec_config = MBartConfig.from_json_file(opt.dec_config_file)

            decoder = MBartDecoder(dec_config, opt)
            decoder.embed_tokens.weight = encoder.embed_tokens.weight
            generators[0].linear.weight = encoder.embed_tokens.weight
            encoder.embed_tokens.weight.requires_grad = False
            decoder.embed_tokens.weight.requires_grad = False
            generators[0].linear.bias.requires_grad = False

        elif opt.dec_pretrained_model in ["m2m", "m2m100"]:
            if opt.enc_pretrained_model not in ["m2m", "m2m100"]:
                from pretrain_module.configuration_m2m100 import M2M100Config
            from pretrain_module.modeling_m2m100 import M2M100Decoder

            dec_config = M2M100Config.from_json_file(opt.dec_config_file)

            decoder = M2M100Decoder(dec_config, opt)
            decoder.embed_tokens.weight = encoder.embed_tokens.weight
            generators[0].linear.weight = encoder.embed_tokens.weight
            # encoder.embed_tokens.weight.requires_grad = False
            # decoder.embed_tokens.weight.requires_grad = False
            # generators[0].linear.bias.requires_grad = False

        elif opt.dec_pretrained_model in ["deltalm"]:
            from onmt.models.deltalm.deltalm import DeltaLMDecoder
            deltalm_config = json_to_namespace(opt.dec_config_file)
            embedding_tgt = nn.Embedding(dicts['tgt'].size(),
                                         deltalm_config.decoder_embed_dim,
                                         padding_idx=constants.TGT_PAD)
            decoder = DeltaLMDecoder(deltalm_config, embedding_tgt)
            # if opt.enc_pretrained_model not in ["deltalm"]:
            #     from pretrain_module.configuration_deltalm import DeltaLMConfig
            # from pretrain_module.modeling_deltalm import DeltaLMDecoder
            #
            # dec_config = DeltaLMConfig.from_json_file(opt.dec_config_file)
            #
            # decoder = DeltaLMDecoder(dec_config, opt)
            # share all embeddings
            decoder.embed_tokens.weight = encoder.embed_tokens.weight
            generators[0].linear.weight = encoder.embed_tokens.weight

            if opt.freeze_embedding:
                decoder.embed_tokens.weight.requires_grad = False
            # generators[0].linear.bias.requires_grad = False

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
        elif opt.dec_pretrained_model:
            print("  Loading weights for decoder from: \n", opt.dec_state_dict)
            dec_model_state_dict = torch.load(opt.dec_state_dict, map_location="cpu")

            if opt.dec_pretrained_model not in ["deltalm"]:
                decoder.from_pretrained(state_dict=dec_model_state_dict,
                                        model=decoder,
                                        output_loading_info=opt.verbose,
                                        model_prefix=opt.dec_pretrained_model
                                        )

        encoder.enc_pretrained_model = opt.enc_pretrained_model
        decoder.dec_pretrained_model = opt.dec_pretrained_model
        print(encoder.enc_pretrained_model, decoder.dec_pretrained_model)

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
            # pass
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
            # pass
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

    if opt.model not in ["pretrain_transformer", "wav2vec2_transformer", "wav2vec2_bert", "wav2vec2"]:
        print('[INFO] Initializing entire model parameters')
        model.apply(weights_init)
    elif opt.model in ['wav2vec2_transformer']:
        print('[INFO] Initializing only decoder parameters')
        model.decoder.apply(weights_init)
    elif opt.model in ['wav2vec2_bert']:
        print("[INFO] Both encoder and decoder are using pretrained weights")
        # freeze the embedding parameters?
    else:
        if opt.enc_pretrained_model and not opt.dec_pretrained_model:
            print('[INFO] Initializing only decoder parameters')
            model.decoder.apply(weights_init)
        if not opt.enc_pretrained_model and opt.dec_pretrained_model:
            print('[INFO] Initializing only encoder parameters')
            model.encoder.apply(weights_init)

    if hasattr(model, 'decoder'):
        if not opt.dec_pretrained_model:
            model.decoder.word_lut.apply(weights_init)
    else:
        if hasattr(model, 'tgt_embedding'):
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

    def convert_fast_attention(m, name):

        def convert(m_):
            classname = m_.__class__.__name__
            if classname.find('MultiheadAttention') != -1:
                m_.convert_fast_attention()
            elif classname.find('MBartAttention') != -1:
                m_.convert_fast_attention()
            elif classname.find('MBartCrossAttention') != -1:
                m_.convert_fast_attention()

        m.apply(convert)

    convert_fast_attention(model, "Transformer")


def optimize_model_test(model):
    """
    Used to potentially upgrade the components with more optimized counterparts in the future
    """

    pass


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
