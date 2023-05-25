import json
import sys

config = dict()

config["encoder_embed_dim"] = 1024 # or 1024?
config["encoder_mssm_num_stacks"] = 1 # change this to stack more s4
config["encoder_mssm_hidden_dim"] = 512
config["encoder_mssm_num_heads"] = 8
config["encoder_mssm_activation"] = "gelu"
config["encoder_mssm_scale"] = 0.5
config["encoder_mssm_maxlen"] = 1024
config["encoder_mssm_timestep_min"] = 0.01
config["encoder_mssm_timestep_max"] = 0.16
config["encoder_mssm_dropout"] = 0.1
config["dropout"] = 0.1
config["activation_fn"] = "gelu"
config["encoder_ffn_embed_dim"] = 4096
config["relu_dropout"] = 0.1

# DECODER SSM ADDED LATER

file_name = sys.argv[1] if len(sys.argv) >= 2 else "ssm_config.json"
out_file = open(file_name, 'w')

json.dump(config, out_file, indent=4)

out_file.close()
