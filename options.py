import argparse


def make_parser(parser):
    # Data options
    parser.add_argument('-data', required=True,
                        help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('-data_format', required=False, default='raw',
                        help='Default data format: raw')

    parser.add_argument('-multi_dataset', action='store_true',
                        help='Reading multiple datasets (sharing the same dictionary)')

    parser.add_argument('-patch_vocab_multiplier', type=int, default=1,
                        help='Pad vocab so that the size divides by this multiplier')
    parser.add_argument('-src_align_right', action="store_true",
                        help="""Aligning the source sentences to the right (default=left for Transformer)""")
    parser.add_argument('-buffer_size', type=int, default=16,
                        help='The iterator fills the data buffer with this size')
    parser.add_argument('-num_workers', type=int, default=0,
                        help='Number of extra workers for data fetching. 0=uses the main process. ')
    parser.add_argument('-pin_memory', action="store_true",
                        help='The data loader pins memory into the GPU to reduce the bottleneck between GPU-CPU')
    parser.add_argument('-memory_profiling', action="store_true",
                        help='Analyze memory consumption for the model')

    parser.add_argument('-bayes_by_backprop', action='store_true',
                        help="""Using Bayes-By-Backprop models in training""")
    parser.add_argument('-neg_log_sigma1', type=float, default=0,
                        help='Coefficient for the KL divergence term')
    parser.add_argument('-neg_log_sigma2', type=float, default=6,
                        help='Coefficient for the KL divergence term')
    parser.add_argument('-prior_pi', type=float, default=0.5,
                        help='Coefficient for the KL divergence term')

    # MODEL UTIL
    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add_argument('-load_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""")
    parser.add_argument('-load_encoder_from', default='', type=str,
                        help="""Load encoder weight from a pretrained model.""")
    parser.add_argument('-load_decoder_from', default='', type=str,
                        help="""Load encoder weight from a pretrained model.""")
    parser.add_argument('-streaming', action='store_true',
                        help="""Using streaming in training""")
    parser.add_argument('-stream_context', default='global', type=str,
                        help="""Using streaming in training""")

    # MODEL CONFIG
    parser.add_argument('-model', default='transformer',
                        help="Translation model. [transformer|relative_transformer  ]")
    parser.add_argument('-layers', type=int, default=2,
                        help='Number of layers in the Transformer encoder/decoder')
    parser.add_argument('-encoder_layers', type=int, default=-1,
                        help='Number of layers in the LSTM encoder if different')
    parser.add_argument('-max_pos_length', type=int, default=128,
                        help='Maximum distance length for relative self-attention')
    parser.add_argument('-learnable_position_encoding', action='store_true',
                        help="""Use embeddings as learnable position encoding.""")
    parser.add_argument('-fix_norm_output_embedding', action='store_true',
                        help="""Normalize the output embedding""")

    parser.add_argument('-asynchronous', action='store_true',
                        help="""Different attention values for past and future""")
    parser.add_argument('-nce_noise', type=int, default=0,
                        help="""Use noise contrastive estimation for the output layer. 
                        Default=0 (full softmax), increase to 100 to use 100 noise samples.""")
    parser.add_argument('-unidirectional', action='store_true',
                        help="""Unidirectional encoder""")
    parser.add_argument('-reconstruct', action='store_true',
                        help='Apply reconstruction with an additional decoder')
    parser.add_argument('-mirror_loss', action='store_true',
                        help='Using mirror loss')

    parser.add_argument('-universal', action='store_true',
                        help='Using one layer universally (recurrent)')
    parser.add_argument('-act', action='store_true',
                        help='Using ACT for Universal models (TODO)')

    # Transforer Model options
    parser.add_argument('-use_language_embedding', action='store_true',
                        help="""Language embedding to add into the word embeddings""")
    parser.add_argument('-language_embedding_type', default='sum', type=str,
                        help="""Language embedding combination type: sum|concat. (Concat uses more parameters)""")
    parser.add_argument('-model_size', type=int, default=512,
                        help='Size of embedding / transformer hidden')
    parser.add_argument('-inner_size', type=int, default=2048,
                        help='Size of inner feed forward layer')
    parser.add_argument('-attribute_size', type=int, default=1,
                        help='Number of attributes')
    parser.add_argument('-n_heads', type=int, default=8,
                        help='Number of heads for multi-head attention')
    parser.add_argument('-checkpointing', type=int, default=0,
                        help='Number of checkpointed layers in the Transformer')
    parser.add_argument('-attn_dropout', type=float, default=0.1,
                        help='Dropout probability; applied on multi-head attention.')
    parser.add_argument('-emb_dropout', type=float, default=0.1,
                        help='Dropout probability; applied on top of embedding.')
    parser.add_argument('-variational_dropout', action='store_true',
                        help='Apply variational dropout (same network per timestep)')
    parser.add_argument('-weight_norm', action='store_true',
                        help='Apply weight normalization on linear modules')
    parser.add_argument('-death_rate', type=float, default=0.0,
                        help='Stochastic layer death rate')
    parser.add_argument('-activation_layer', default='linear_relu_linear', type=str,
                        help='The activation layer in each transformer block '
                             'linear_relu_linear|linear_swish_linear|maxout')
    parser.add_argument('-time', default='positional_encoding', type=str,
                        help='Type of time representation positional_encoding|gru|lstm')
    parser.add_argument('-version', type=float, default=1.0,
                        help='Deprecated.')
    parser.add_argument('-residual_type', default='regular',
                        help='Type of residual type. regular|gated')
    parser.add_argument('-adaptive', type=str, default='shared',
                        help='Universal adaptive layer. universal=UniversalTF|shared=factorized|unshared')
    # Optimization options
    parser.add_argument('-encoder_type', default='text',
                        help="Type of encoder to use. Options are [text|img].")
    parser.add_argument('-input_size', type=int, default=2048,
                        help='Size of input features')
    parser.add_argument('-init', default='normal',
                        help="How to init the weight. normal or uniform/xavier.")
    parser.add_argument('-init_embedding', default='normal',
                        help="How to init the embedding matrices. Xavier or Normal.")
    parser.add_argument('-batch_size_words', type=int, default=2048,
                        help='Maximum batch size in word dimension')
    parser.add_argument('-batch_size_sents', type=int, default=128,
                        help='Maximum number of sentences in a batch')

    parser.add_argument('-batch_size_update', type=int, default=-1,
                        help='Maximum number of words per update')
    parser.add_argument('-update_frequency', type=int, default=1,
                        help='Maximum number of batches per update (will override the batch_size_update')
    parser.add_argument('-batch_size_multiplier', type=int, default=1,
                        help='Maximum number of words per update')
    parser.add_argument('-max_position_length', type=int, default=1024,
                        help='Maximum length for positional embedding')
    parser.add_argument('-max_memory_size', type=int, default=1024,
                        help='Maximum memory size for buffering in transformer XL')
    parser.add_argument('-extra_context_size', type=int, default=32,
                        help='Extra context size in transformer Xl')
    parser.add_argument('-epochs', type=int, default=13,
                        help='Number of training epochs')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init)""")
    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-max_grad_norm', type=float, default=0,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")

    # Dropout
    parser.add_argument('-dropout', type=float, default=0.3,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-word_dropout', type=float, default=0.0,
                        help='Dropout probability; applied on embedding indices.')
    parser.add_argument('-switchout', type=float, default=0.0,
                        help='Switchout algorithm')

    # Loss function
    parser.add_argument('-label_smoothing', type=float, default=0.0,
                        help='Label smoothing value for loss functions.')
    parser.add_argument('-scheduled_sampling_rate', type=float, default=0.0,
                        help='Scheduled sampling rate.')

    parser.add_argument('-curriculum', type=int, default=-1,
                        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""")
    parser.add_argument('-normalize_gradient', action="store_true",
                        help="""Normalize the gradients by number of tokens before updates""")
    # learning rate
    parser.add_argument('-learning_rate', type=float, default=1.0,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=1,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=99999,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-warmup_steps', type=int, default=4096,
                        help="""Number of steps to increase the lr in noam""")
    parser.add_argument('-max_steps', type=int, default=100000,
                        help="""Number of steps to train the model""")
    parser.add_argument('-noam_step_interval', type=int, default=1,
                        help="""How many steps before updating the parameters""")
    parser.add_argument('-max_step', type=int, default=40000,
                        help="""How many steps before updating the parameters""")
    parser.add_argument('-reset_optim', action='store_true',
                        help='Reset the optimizer running variables')
    parser.add_argument('-beta1', type=float, default=0.9,
                        help="""beta_1 value for adam""")
    parser.add_argument('-beta2', type=float, default=0.98,
                        help="""beta_2 value for adam""")
    parser.add_argument('-weight_decay', type=float, default=0.0,
                        help="""weight decay (L2 penalty)""")
    parser.add_argument('-amsgrad', action='store_true',
                        help='Using AMSGRad for adam')
    parser.add_argument('-update_method', default='regular',
                        help="Type of update rule to use. Options are [regular|noam].")
    # pretrained word vectors
    parser.add_argument('-tie_weights', action='store_true',
                        help='Tie the weights of the encoder and decoder layer')
    parser.add_argument('-experimental', action='store_true',
                        help='Set the model into the experimental mode (trying unverified features)')
    parser.add_argument('-join_embedding', action='store_true',
                        help='Jointly train the embedding of encoder and decoder in one weight')
    parser.add_argument('-add_position_encoding', action='store_true',
                        help='Adding pos encodings to embedding (like Transformer)')
    parser.add_argument('-batch_ensemble', type=int, default=0,
                        help='To use batch ensemble algorithm')

    # GPU
    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-fp16', action='store_true',
                        help='Use half precision training')
    parser.add_argument('-fp16_mixed', action='store_true',
                        help='Use mixed half precision training. fp16 must be enabled.')
    parser.add_argument('-seed', default=-1, type=int,
                        help="Seed for deterministic runs.")

    parser.add_argument('-log_interval', type=int, default=100,
                        help="Print stats at this interval.")
    parser.add_argument('-save_every', type=int, default=-1,
                        help="Save every this interval.")
    parser.add_argument('-keep_save_files', type=int, default=5,
                        help="Save every this interval.")
    parser.add_argument('-copy_generator', action='store_true',
                        help='Use the copy_generator')
    parser.add_argument('-verbose', action='store_true',
                        help='Show more information about training (for Nerds)')
    # FAST IMPLEMENTATION
    parser.add_argument('-fast_xentropy', action="store_true",
                        help="""Fast cross entropy loss""")
    parser.add_argument('-fast_xattention', action="store_true",
                        help="""Fast cross attention between encoder decoder""")
    parser.add_argument('-fast_self_attention', action="store_true",
                        help="""Fast self attention between encoder decoder""")
    parser.add_argument('-fast_feed_forward', action="store_true",
                        help="""Fast cross attention between encoder decoder""")
    parser.add_argument('-macaron', action='store_true',
                        help='Macaron style network with 2 FFN per block.')
    parser.add_argument('-fused_ffn', action="store_true",
                        help="""Fast feedforward""")

    # for FUSION
    parser.add_argument('-lm_checkpoint', default='', type=str,
                        help="""If training from a checkpoint then this is the
                            path to the pretrained model.""")
    parser.add_argument('-fusion', action='store_true',
                        help='Use fusion training with language model')
    parser.add_argument('-lm_seq_length', type=int, default=128,
                        help='Sequence length for the language model')

    # for Speech
    parser.add_argument('-reshape_speech', type=int, default=0,
                        help="Reshaping the speech data (0 is ignored, done at preprocessing).")
    parser.add_argument('-concat', type=int, default=4,
                        help="Concatenate frames to downsample.")
    parser.add_argument('-input_feature_size', type=int, default=40,
                        help="Input feature size.")
    parser.add_argument('-augment_speech', action='store_true',
                        help='Use f/t augmentation for speech')
    parser.add_argument('-upsampling', action='store_true',
                        help='In case the data is downsampled during preprocess. This option will upsample the '
                             'samples again')
    parser.add_argument('-cnn_downsampling', action='store_true',
                        help='Use CNN for downsampling instead of reshaping')
    parser.add_argument('-zero_encoder', action='store_true',
                        help='Zero-out encoders during training')
    parser.add_argument('-ctc_loss', type=float, default=0.0,
                        help='CTC Loss as additional loss function with this weight')
    parser.add_argument('-lfv_multilingual', action='store_true',
                        help='Use multilingual language identifier to get LFV for each language')
    parser.add_argument('-bottleneck_size', type=int, default=64,
                        help="Bottleneck size for the LFV vector).")
    parser.add_argument('-conv_kernel', type=int, default=31,
                        help="Kernels for convolution in conformer).")
    parser.add_argument('-no_batch_norm', action='store_true',
                        help="Remove Batch Norm to avoid NaN errors that can happen with spec augmentation.).")
    parser.add_argument('-depthwise_conv', action='store_true',
                        help='Use depthwise convolution in the encoder block')
    parser.add_argument('-no_ffn', action='store_true',
                        help='No feedforward network in the speech encoder')

    parser.add_argument('-multilingual_factorized_weights', action='store_true',
                        help='Factorize the weights in the model for multilingual')
    parser.add_argument('-mfw_rank', type=int, default=1,
                        help="Rank of the mfw vectors.")
    parser.add_argument('-mfw_multiplicative', action='store_true',
                        help='Use another multiplicative weights W = W^ * M + A')

    parser.add_argument('-multilingual_partitioned_weights', action='store_true',
                        help='Partition the weights in the multilingual models')
    parser.add_argument('-mpw_factor_size', type=int, default=8,
                        help="Size of the language factor vector")
    parser.add_argument('-multilingual_layer_norm', action='store_true',
                        help='New norm for each language')
    parser.add_argument('-multilingual_linear_projection', action='store_true',
                        help='New linear projection for each language')
    parser.add_argument('-sub_encoder', type=int, default=4,
                        help='New linear projection for each language')
    parser.add_argument('-weight_drop', type=float, default=0.0,
                        help='dropout rate for the main weights of the MFW model')

    # for Reformer
    # parser.add_argument('-lsh_src_attention', action='store_true',
    #                     help='Using LSH for source attention')
    # parser.add_argument('-chunk_length', type=int, default=32,
    #                     help="Length of chunk which attends to itself in LSHSelfAttention.")
    # parser.add_argument('-lsh_num_chunks_after', type=int, default=0,
    #                     help="Length of chunk which attends to itself in LSHSelfAttention.")
    # parser.add_argument('-lsh_num_chunks_before', type=int, default=1,
    #                     help="Length of chunk which attends to itself in LSHSelfAttention.")
    # parser.add_argument('-num_hashes', type=int, default=4,
    #                     help="Number of hasing rounds.")

    # for Reversible Transformer
    parser.add_argument('-src_reversible', action='store_true',
                        help='Using reversible models for encoder')
    parser.add_argument('-tgt_reversible', action='store_true',
                        help='Using reversible models for decoder')

    parser.add_argument('-debugging', action='store_true',
                        help='Using reversible models for decoder')
    parser.add_argument('-master_addr', default='localhost', type=str,
                        help="""""")
    parser.add_argument('-master_port', default='8888', type=str,
                        help="""""")
    return parser


def backward_compatible(opt):
    # FOR BACKWARD COMPATIBILITY

    if not hasattr(opt, 'model'):
        opt.model = 'recurrent'

    if not hasattr(opt, 'layer_norm'):
        opt.layer_norm = 'slow'

    if not hasattr(opt, 'attention_out'):
        opt.attention_out = 'default'

    if not hasattr(opt, 'residual_type'):
        opt.residual_type = 'regular'

    if not hasattr(opt, 'input_size'):
        opt.input_size = 40

    if not hasattr(opt, 'init_embedding'):
        opt.init_embedding = 'normal'

    if not hasattr(opt, 'ctc_loss'):
        opt.ctc_loss = 0

    if not hasattr(opt, 'encoder_layers'):
        opt.encoder_layers = -1

    if not hasattr(opt, 'fusion'):
        opt.fusion = False

    if not hasattr(opt, 'cnn_downsampling'):
        opt.cnn_downsampling = False

    if not hasattr(opt, 'switchout'):
        opt.switchout = 0.0

    if not hasattr(opt, 'variational_dropout'):
        opt.variational_dropout = False

    if not hasattr(opt, 'copy_generator'):
        opt.copy_generator = False

    if not hasattr(opt, 'upsampling'):
        opt.upsampling = False

    if not hasattr(opt, 'double_position'):
        opt.double_position = False

    if not hasattr(opt, 'max_pos_length'):
        opt.max_pos_length = 0

    if not hasattr(opt, 'learnable_position_encoding'):
        opt.learnable_position_encoding = False

    if not hasattr(opt, 'use_language_embedding'):
        opt.use_language_embedding = False

    if not hasattr(opt, 'language_embedding_type'):
        opt.language_embedding_type = "sum"

    if not hasattr(opt, 'asynchronous'):
        opt.asynchronous = False

    if not hasattr(opt, 'bidirectional'):
        opt.bidirectional = False

    if not hasattr(opt, 'fix_norm_output_embedding'):
        opt.fix_norm_output_embedding = False

    if not hasattr(opt, 'mirror_loss'):
        opt.mirror_loss = False

    if not hasattr(opt, 'max_memory_size'):
        opt.max_memory_size = 0

    if not hasattr(opt, 'stream_context'):
        opt.stream_context = 'local'

    if not hasattr(opt, 'extra_context_size'):
        opt.extra_context_size = 0

    if opt.model == 'relative_unified_transformer' and not opt.src_align_right:
        print(" !!! Warning: model %s requires source sentences aligned to the right (-src_align_right)" % opt.model)

    if not hasattr(opt, 'experimental'):
        opt.experimental = False

    if not hasattr(opt, 'reconstruct'):
        opt.reconstruct = False

    if not hasattr(opt, 'unidirectional'):
        opt.unidirectional = False

    if not hasattr(opt, 'lsh_src_attention'):
        opt.lsh_src_attention = False

    if not hasattr(opt, 'src_reversible'):
        opt.src_reversible = False

    if not hasattr(opt, 'tgt_reversible'):
        opt.tgt_reversible = False

    if not hasattr(opt, 'fast_xentropy'):
        opt.fast_xentropy = False

    if not hasattr(opt, 'fast_xattention'):
        opt.fast_xattention = False

    if not hasattr(opt, 'fast_self_attention'):
        opt.fast_self_attention = False

    if not hasattr(opt, 'fast_feed_forward'):
        opt.fast_feed_forward = False

    if not hasattr(opt, 'fused_ffn'):
        opt.fused_ffn = False

    if not hasattr(opt, 'concat'):
        opt.concat = 4

    if not hasattr(opt, 'input_feature_size'):
        opt.input_feature_size = 40

    if not hasattr(opt, 'bayes_by_backprop'):
        opt.bayes_by_backprop = False

    if not hasattr(opt, 'add_position_encoding'):
        opt.add_position_encoding = False

    if not hasattr(opt, 'batch_ensemble'):
        opt.batch_ensemble = 0

    if not hasattr(opt, 'multilingual_factorized_weights'):
        opt.multilingual_factorized_weights = False

    if not hasattr(opt, 'mfw_rank'):
        opt.mfw_rank = 1

    if not hasattr(opt, 'lfv_multilingual'):
        opt.lfv_multilingual = False

    if not hasattr(opt, 'nce_noise'):
        opt.nce_noise = 0

    if not hasattr(opt, 'mfw_multiplicative'):
        opt.mfw_multiplicative = False

    if not hasattr(opt, 'macaron'):
        opt.macaron = False

    if not hasattr(opt, 'depthwise_conv'):
        opt.depthwise_conv = False

    if not hasattr(opt, 'fused_ffn'):
        opt.fused_ffn = False

    if not hasattr(opt, 'no_batch_norm'):
        opt.no_batch_norm = False

    if not hasattr(opt, 'no_ffn'):
        opt.no_ffn = False

    if not hasattr(opt, 'multilingual_partitioned_weights'):
        opt.multilingual_partitioned_weights = False

    if not hasattr(opt, 'mpw_factor_size'):
        opt.mpw_factor_size = 1

    if not hasattr(opt, 'multilingual_layer_norm'):
        opt.multilingual_layer_norm = False

    if not hasattr(opt, 'multilingual_linear_projection'):
        opt.multilingual_linear_projection = False

    if not hasattr(opt, 'weight_drop'):
        opt.weight_drop = 0.0

    return opt
