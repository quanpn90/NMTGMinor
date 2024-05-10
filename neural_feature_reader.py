# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import soundfile as sf
import torch.nn.functional as F
import tqdm
import torchaudio

from onmt.data.audio_utils import safe_readaudio, wav_to_fmel


class FeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer, max_chunk=1600000, use_cuda=True, fp16=False, bf16=False,
                 model_path="w2vbert-conformer_shaw.pt", sample_rate=16000):

        # first we have to load the model
        # lets try to load the wav2vec-bert model
        # (
        #     model,
        #     cfg,
        #     task,
        # ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        #     [checkpoint_path]
        # )

        from onmt.models.speech_recognizer.w2v_bert.config import conformer_shaw_600m
        from onmt.models.speech_recognizer.w2v_bert.builder import create_conformer_shaw_model
        config = conformer_shaw_600m()

        self.model = create_conformer_shaw_model(config)

        if len(model_path) > 0:
            cpt = torch.load(model_path, map_location=torch.device('cpu'))
            weights = cpt['model']
            print("[INFO] Loaded pretrained-w2vbert model")
            self.model.load_state_dict(weights)

        self.model.eval()
        self.layer = layer
        self.max_chunk = max_chunk
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

        self.fp16 = fp16
        self.bf16 = bf16

        self.sample_rate = sample_rate

    def read_audio(self, path, ref_len=None, channel_id=None):
        # wav, sr = sf.read(path)
        # if channel_id is not None:
        #     assert wav.ndim == 2, \
        #         f"Expected stereo input when channel_id is given ({path})"
        #     assert channel_id in [1, 2], \
        #         "channel_id is expected to be in [1, 2]"
        #     wav = wav[:, channel_id-1]
        # if wav.ndim == 2:
        #     wav = wav.mean(-1)
        # assert wav.ndim == 1, wav.ndim
        # assert sr == self.sample_rate, sr
        # if ref_len is not None and abs(ref_len - len(wav)) > 160:
        #     print(f"ref {ref_len} != read {len(wav)} ({path})")

        # wav = torchaudio.load(path)
        wav = safe_readaudio(path)
        wav = wav_to_fmel(wav, num_mel_bin=80)

        return wav

    def get_feats(self, file_path, ref_len=None, channel_id=None):
        x = self.read_audio(file_path, ref_len, channel_id)
        with torch.no_grad():
            x = x.float()

            # TODO: use torch amp.autocast here
            if self.use_cuda:
                x = x.cuda()

            # batch size 1
            # x = x.view(1, -1)
            x = x.unsqueeze(0)

            feat = []
            # for start in range(0, x.size(1), self.max_chunk):
            #     x_chunk = x[:, start: start + self.max_chunk]
            #     # feat_chunk, _ = self.model.extract_features(
            #     #     source=x_chunk,
            #     #     padding_mask=None,
            #     #     mask=False,
            #     #     output_layer=self.layer,
            #     # )

                # self.model.forward(x_chunk, padding_mask=None, layer=self.layer)
                #
                # feat.append(feat_chunk)
            feat, _ = self.model.forward(x, padding_mask=None, layer=self.layer)
        return feat.squeeze(0)

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Learn K-means clustering over acoustic features."
    )

    # Features arguments
    parser.add_argument(
        "--in_features_path", type=str, default=None, help="Features file path"
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["logmel", "hubert", "w2v2", "cpc"],
        default=None,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--out_features_path",
        type=str,
        default=None,
        help="Features file path to write to",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Pretrained acoustic model checkpoint",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--sample_pct",
        type=float,
        help="Percent data to use for K-means training",
        default=0.1,
    )

    # K-means arguments
    parser.add_argument(
        "--num_clusters", type=int, help="Number of clusters", default=100
    )
    parser.add_argument("--init", default="k-means++")
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Maximum number of iterations for K-means training",
        default=150,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for K-means training",
        default=10000,
    )
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.5, type=float)
    parser.add_argument(
        "--out_kmeans_model_path",
        type=str,
        required=False,
        help="Path to save K-means model",
    )

    # Leftovers
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to use for K-means training",
        default=1369,
    )

    return parser


def get_kmeans_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
    random_state,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        tol=tol,
        max_no_improvement=max_no_improvement,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
        random_state=random_state,
        verbose=1,
        compute_labels=True,
        init_size=None,
    )


def train_kmeans(kmeans_model, features_batch):
    start_time = time.time()
    kmeans_model.fit(features_batch)
    time_taken = round((time.time() - start_time) // 60, 2)
    return kmeans_model, time_taken





def get_feature_iterator(
    checkpoint_path, layer, manifest_path, sample_pct, channel_id,
    fp16=False, bf16=False, model_path="w2vbert-conformer_shaw.pt", sample_rate=16000
):
    feature_reader = FeatureReader(checkpoint_path,
                                   layer,
                                   max_chunk=1600000,
                                   use_cuda=True,
                                   fp16=fp16,
                                   bf16=bf16,
                                   model_path=model_path, sample_rate=sample_rate)

    print("Feature extract successfully created")

    with open(manifest_path, "r") as fp:
        lines = fp.readlines()
        # root = lines.pop(0).strip()
        file_path_list = [
            line.split()[1]
            for line in lines
            if len(line) > 0
        ]
        if sample_pct < 1.0:
            file_path_list = random.sample(
                file_path_list, int(sample_pct * len(file_path_list))
            )
        num_files = len(file_path_list)
        reader = feature_reader

        def iterate():
            for file_path in file_path_list:
                feats = reader.get_feats(file_path, channel_id=channel_id)
                yield feats.cpu().numpy()

    return iterate, num_files


def get_features(
        checkpoint_path, layer, manifest_path, sample_pct, channel_id,
        fp16=False, bf16=False, model_path="w2vbert-conformer_shaw.pt", sample_rate=16000,
        flatten=True,
):
    generator, num_files = get_feature_iterator(checkpoint_path, layer, manifest_path, sample_pct, channel_id,
                                                fp16=fp16, bf16=bf16, model_path=model_path, sample_rate=sample_rate
    )
    iterator = generator()

    features_list = []
    for features in tqdm.tqdm(iterator, total=num_files):
        features_list.append(features)

    # Explicit clean up
    del iterator
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    if flatten:
        return np.concatenate(features_list)

    return features_list


def main(args, logger):
    # Features loading/extraction for K-means
    # if args.in_features_path:
    #     # Feature loading
    #     logger.info(f"Loading features from {args.in_features_path}...")
    #     features_batch = np.load(args.in_features_path, allow_pickle=True)
    # else:
    #     # Feature extraction
    #     logger.info(f"Extracting {args.feature_type} acoustic features...")
    #     features_batch = (
    #         get_features(
    #             feature_type=args.feature_type,
    #             checkpoint_path=args.checkpoint_path,
    #             layer=args.layer,
    #             manifest_path=args.manifest_path,
    #             sample_pct=args.sample_pct,
    #             flatten=True,
    #         )
    #         if not args.out_features_path
    #         else get_and_dump_features(
    #             feature_type=args.feature_type,
    #             checkpoint_path=args.checkpoint_path,
    #             layer=args.layer,
    #             manifest_path=args.manifest_path,
    #             sample_pct=args.sample_pct,
    #             flatten=True,
    #             out_features_path=args.out_features_path,
    #         )
    #     )
    #     if args.out_features_path:
    #         logger.info(
    #             f"Saved extracted features at {args.out_features_path}"
    #         )

    logger.info(f"Extracting w2vbert acoustic features...")

    features_batch = (
        get_features(
            checkpoint_path=args.checkpoint_path,
            layer=args.layer,
            manifest_path=args.manifest_path,
            channel_id=None,
            sample_pct=args.sample_pct,
            flatten=True,
            fp16=True,
            bf16=False,
            model_path="w2vbert-conformer_shaw.pt",
            sample_rate=16000,
        )
    )

    # logger.info(f"Features shape = {features_batch.shape}\n")

    # Learn and save K-means model
    # kmeans_model = get_kmeans_model(
    #     n_clusters=args.num_clusters,
    #     init=args.init,
    #     max_iter=args.max_iter,
    #     batch_size=args.batch_size,
    #     tol=args.tol,
    #     max_no_improvement=args.max_no_improvement,
    #     n_init=args.n_init,
    #     reassignment_ratio=args.reassignment_ratio,
    #     random_state=args.seed,
    # )
    # logger.info("Starting k-means training...")
    # kmeans_model, time_taken = train_kmeans(
    #     kmeans_model=kmeans_model, features_batch=features_batch
    # )
    # logger.info(f"...done k-means training in {time_taken} minutes")
    # inertia = -kmeans_model.score(features_batch) / len(features_batch)
    # logger.info(f"Total intertia: {round(inertia, 2)}\n")
    #
    # logger.info(f"Saving k-means model to {args.out_kmeans_model_path}")
    # os.makedirs(os.path.dirname(args.out_kmeans_model_path), exist_ok=True)
    # joblib.dump(kmeans_model, open(args.out_kmeans_model_path, "wb"))

    return features_batch


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)