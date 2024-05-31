import torch
import sys

input_file = sys.argv[1]

kmeans_size = int(sys.argv[2])  if len(sys.argv) > 2 else 5947

cpt = torch.load(input_file)

print(cpt.keys())

weights = cpt['weights']

base_vocab_size = 250053 #5947

embedding_keys = ["tgt_embedding.embed_tokens.weight",
                  "src_embedding.embed_tokens.weight",
                  ]

for key in weights:

    if key in embedding_keys:
        print("checking embedding layer: ", key)
        embedding_weights = weights[key]

        print(embedding_weights.size())

        # allocate new embeddings
        embed_size = embedding_weights.size(-1)
        new_embedding = embedding_weights.new_zeros(base_vocab_size + kmeans_size, embed_size)
        torch.nn.init.normal_(new_embedding)

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

        weights[key] = new_embedding

cpt['weights'] = weights


output_file = input_file + ".extend"

print("Saving new deltalm weights to %s " % output_file)
torch.save(cpt, output_file)
