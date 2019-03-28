import argparse

def make_parser(parser):
    
    
    # Data options
    parser.add_argument('-data', required=True,
                        help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('-data_format', required=False, default='bin',
                        help='Default data format: raw')
    parser.add_argument('-sort_by_target', action='store_true',
                        help='Training data sorted by target')                    
    parser.add_argument('-pad_count', action='store_true',
                        help='Training data sorted by target')                    
    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add_argument('-load_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""")
    parser.add_argument('-model', default='recurrent',
                        help="Optimization method. [recurrent|transformer|stochastic_transformer]")
    parser.add_argument('-layers', type=int, default=2,
                        help='Number of layers in the LSTM encoder/decoder')                   
    # Recurrent Model options
    parser.add_argument('-rnn_size', type=int, default=512,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int, default=512,
                        help='Word embedding sizes')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")

    parser.add_argument('-loss_function', type=int, default=0,
                        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""")

    # Transforer Model options
    parser.add_argument('-model_size', type=int, default=512,
        help='Size of embedding / transformer hidden')      
    parser.add_argument('-inner_size', type=int, default=2048,
        help='Size of inner feed forward layer')  
    parser.add_argument('-n_heads', type=int, default=8,
        help='Number of heads for multi-head attention')
    parser.add_argument('-n_encoder_heads', type=int, default=1,
                        help='Number of heads for simplified encoder ')
    parser.add_argument('-checkpointing', type=int, default=0,
        help='Number of checkpointed layers in the Transformer') 
    parser.add_argument('-attn_dropout', type=float, default=0.1,
                        help='Dropout probability; applied on multi-head attention.')   
    parser.add_argument('-emb_dropout', type=float, default=0.1,
                        help='Dropout probability; applied on top of embedding.')    
    parser.add_argument('-residual_dropout', type=float, default=0.2,
                        help='Dropout probability; applied on residual connection.')    
    parser.add_argument('-weight_norm', action='store_true',
                      help='Apply weight normalization on linear modules')
    parser.add_argument('-layer_norm', default='fast',
                      help='Layer normalization type')
    parser.add_argument('-death_rate', type=float, default=0.5,
                        help='Stochastic layer death rate')  
    parser.add_argument('-death_type', type=str, default='linear_decay',
                        help='Stochastic layer death type: linear decay or uniform')  
    parser.add_argument('-activation_layer', default='linear_relu_linear', type=str,
                        help='The activation layer in each transformer block')                        
    parser.add_argument('-time', default='positional_encoding', type=str,
                        help='Type of time representation positional_encoding|gru|lstm')                        
    parser.add_argument('-version', type=float, default=1.0,
                        help='Transformer version. 1.0 = Google type | 2.0 is different')                    
    parser.add_argument('-attention_out', default='default',
                      help='Type of attention out. default|combine')
    parser.add_argument('-residual_type', default='regular',
                      help='Type of residual type. regular|gated')
    # Optimization options
    parser.add_argument('-encoder_type', default='text',
                        help="Type of encoder to use. Options are [text|img].")
    parser.add_argument('-init_embedding', default='normal',
                        help="How to init the embedding matrices. Xavier or Normal.")
    parser.add_argument('-batch_size_words', type=int, default=2048,
                        help='Maximum batch size in word dimension')
    parser.add_argument('-batch_size_sents', type=int, default=128,
                        help='Maximum number of sentences in a batch')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""")
    parser.add_argument('-batch_size_update', type=int, default=2048,
                        help='Maximum number of words per update')                    
    parser.add_argument('-batch_size_multiplier', type=int, default=1,
                        help='Maximum number of words per update')                    
    parser.add_argument('-max_position_length', type=int, default=1024,
        help='Maximum length for positional embedding')    

    parser.add_argument('-epochs', type=int, default=13,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init)""")
    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-max_grad_norm', type=float, default=0,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.3,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-word_dropout', type=float, default=0.0,
                        help='Dropout probability; applied on embedding indices.')
    parser.add_argument('-label_smoothing', type=float, default=0.0,
                        help='Label smoothing value for loss functions.')
    parser.add_argument('-scheduled_sampling_rate', type=float, default=0.0,
                        help='Scheduled sampling rate.')
    parser.add_argument('-curriculum', type=int, default=-1,
                        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""")
    parser.add_argument('-extra_shuffle', action="store_true",
                        help="""By default only shuffle mini-batch order; when true,
                        shuffle and re-assign mini-batches""")
    parser.add_argument('-normalize_gradient', action="store_true",
                        help="""Normalize the gradients by number of tokens before updates""")
    parser.add_argument('-virtual_gpu', type=int, default=1,
                        help='Number of virtual gpus. The trainer will try to mimic asynchronous multi-gpu training')
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
    parser.add_argument('-noam_step_interval', type=int, default=1,
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
                        help='Tie the weights between the decoder embeddings and softmax')
    parser.add_argument('-join_embedding', action='store_true',
                        help='Jointly train the embedding of encoder and decoder in one weight matrix')
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")

    # GPU
    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-fp16', action='store_true',
                        help='Use half precision training')     
    parser.add_argument('-fp16_loss_scale', type=float, default=8,
                        help="""Loss scale for fp16 loss (to avoid overflowing in fp16).""")
    parser.add_argument('-seed', default=9999, type=int,
                        help="Seed for deterministic runs.")

    parser.add_argument('-log_interval', type=int, default=100,
                        help="Print stats at this interval.")
    parser.add_argument('-save_every', type=int, default=-1,
                        help="Save every this interval.")
    parser.add_argument('-keep_save_files', type=int, default=5,
                        help="Save every this interval.")

    parser.add_argument('-num_valid_samples', type=int, default=10,
                        help="Number of sampling times during validation.")
    parser.add_argument('-tau', type=float, default=1.0,
                        help="Number of sampling times during validation.")
    
    parser.add_argument('-n_mixtures', type=int, default=10,
                        help="Number of mixtures in moe.")

    parser.add_argument('-share_enc_dec_weights', action='store_true',
                        help='Share the encoder and decoder weights (except for the src attention layer)')

    parser.add_argument('-var_posterior_combine', default='concat',
                        help="Type of combination between source and target for posterior q(z|x, y). Values concat|sum")
    parser.add_argument('-var_ignore_source', action='store_true',
                        help="Ignore source sentence in the decoder (only relying on the latent variable z)")
    parser.add_argument('-var_posterior_share_weight', action='store_true',
                        help="Share weights between posterior")

    parser.add_argument('-var_ignore_first_source_token', action='store_true',
                        help="Share weights between posterior")
    parser.add_argument('-var_ignore_first_target_token', action='store_true',
                        help="Share weights between posterior")
    parser.add_argument('-var_kl_lambda', type=float, default=1.0,  
                        help="""kl coefficient in the loss function""")
    parser.add_argument('-var_latent_dim', type=int, default=64,
                        help="Number of dimensions for the latent variable.")
    parser.add_argument('-var_pooling', default='mean',  
                        help="""pooling operation to summerize one state from a sequence""")
    parser.add_argument('-var_combine_z', default='once',  
                        help="""How to combine z into the transformer: once|all """)
    parser.add_argument('-load_pretrained_nmt', default='', type=str,
                        help="""Load a pretrained from a transformer model""")
    parser.add_argument('-var_not_sampling', action='store_true',
                        help="""Do not sample (use mean)""")
    parser.add_argument('-var_annealing_kl', action='store_true',
                        help="""Annealing the cofficient of kl divergence during training as in Bowman et al, 2016""")
    parser.add_argument('-var_sample_from', default='posterior',
                        help="""The distribution where we sample from. Default is posterior. posterior|prior""")
    parser.add_argument('-var_depth', type=int, default=16,
                        help="Number of recurrent states in the variational layer.")

    parser.add_argument('-l2_coeff', type=float, default=1.0,
                        help="""l2 coefficient in the loss function for multilingual""")
    return parser