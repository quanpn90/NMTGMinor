import torch
import sys
import random

input_file = sys.argv[1]

vocab_file = sys.argv[2]

kmeans_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5946

cpt = torch.load(input_file, map_location=lambda storage, loc: storage)

print(cpt.keys())

weights = cpt['weights']

base_vocab_size = 250054 # 5947

embedding_keys = ["tgt_embedding.embed_tokens.weight",
                  "src_embedding.embed_tokens.weight",
                  ]

for key in weights:

    if key in embedding_keys:
        print("checking embedding layer: ", key)
        embedding_weights = weights[key]

        print(embedding_weights.size())
        var_mean = torch.var_mean(embedding_weights, dim=0)

        var = var_mean[0]
        mean = var_mean[1]

        print("Variation:", var)
        print("Mean:", mean)

        print(var.mean())
        print(mean.mean())

        # allocate new embeddings
        embed_size = embedding_weights.size(-1)
        new_embedding = embedding_weights.new_zeros(base_vocab_size + kmeans_size, embed_size)
        torch.nn.init.normal_(new_embedding, mean=mean.mean().item(), std=var.mean().item())

        print(new_embedding.size())

        new_vocab_size = base_vocab_size + kmeans_size

        # copy the data.
        # the embedding_weights can have size > base_vocab_size
        # (because in deltalm they used many masked tokens to train)
        new_embedding[:base_vocab_size, :].copy_(embedding_weights[:base_vocab_size, :])

        # check
        for i in range(base_vocab_size):

            old_vector = embedding_weights[i]
            new_vector = new_embedding[i]

            assert torch.equal(old_vector, new_vector), "Vector not copied properly"

        # take random vectors to initialize
        for j in range(kmeans_size):

            rand_i = random.randrange(base_vocab_size)
            new_embedding[base_vocab_size + j, :].copy_(new_embedding[rand_i, :])

        weights[key] = new_embedding

cpt['weights'] = weights


output_file = input_file + ".extend"

print("Saving new deltalm weights to %s " % output_file)
torch.save(cpt, output_file)


vocab_data = open(vocab_file, "r")
vocab_writer = open(vocab_file + ".extend", 'w')

c = 0
for line in vocab_data.readlines():

    vocab_writer.write(line)

    c = c + 1

for i in range(kmeans_size):

    new_word = "__" + str(i) + "__"

    new_id = c + i

    new_line = new_word + " " + str(new_id) + "\n"

    vocab_writer.write(new_line)



