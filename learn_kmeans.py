# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import soundfile as sf
import torch.nn.functional as F
import tqdm
import torchaudio
import gc
import os
import torch.multiprocessing as mp

from onmt.data.audio_utils import safe_readaudio, wav_to_fmel


import argparse
import logging
import os
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins

class FeatureReader:

    @staticmethod
    def find_offsets(filename, num_chunks):
        """
        :param filename: string
        :param num_chunks: int
        :return: a list of offsets (positions to start and stop reading)
        """
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def load_feature_single_thread(filename, worker_id, offset, end):
        """
        This function should read in the lines, convert sentences to tensors
        And then finalize into a dataset?
        """

        result = dict()
        data = list()

        count = 0

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)

            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)


            while line:
                if 0 < end < f.tell():
                    break

                file_path = line.split()[1]
                feat = np.load(file_path)['arr_0']
                data.append(feat)

                line = f.readline()

                count += 1
                if count % 10000 == 0:
                    print("[INFO] Thread %d processed %d lines." % (worker_id, count))


        print("[INFO] Thread %d Done." % worker_id)
        result['data'] = data
        result['id'] = worker_id

        return result

    @staticmethod
    def load_features(filename, num_workers=1, verbose=False):

        result = dict()

        for i in range(num_workers):
            result[i] = dict()

        final_result = dict()

        def merge_result(bin_result):
            result[bin_result['id']]['data'] = bin_result['data']

        offsets = FeatureReader.find_offsets(filename, num_workers)

        if num_workers > 1:

            pool = mp.Pool(processes=num_workers)
            mp_results = []

            for worker_id in range(num_workers):
                mp_results.append(pool.apply_async(
                    FeatureReader.load_feature_single_thread,
                    args=(filename, worker_id,
                          offsets[worker_id], offsets[worker_id + 1]),
                ))

            pool.close()
            pool.join()

            for r in mp_results:
                merge_result(r.get())

        else:
            sp_result = BFeatureReader.load_feature_single_thread(filename, 0,
                                                              offsets[0], offsets[1])
            merge_result(sp_result)

        final_result['data'] = list()

        # put the data into the list according the worker indices
        for idx in range(num_workers):
            final_result['data'] += result[idx]['data']

        return final_result['data']



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
        "--data_type",
        type=str,
        choices=["fp16", "fp32", "bf16"],
        default="fp32",
        help="data type (for half-precision)",
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
        default="features",
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

    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers to load features",
        default=1,
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
    manifest_path, sample_pct
):

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

        def iterate():
            for file_path in file_path_list:
                # feats = reader.get_feats(file_path, channel_id=channel_id)
                feat = np.load(file_path)['arr_0']

                yield feat

    return iterate, num_files


def read_features(
        manifest_path, num_workers=1, sample_pct=1.0,
        flatten=True
):
    # generator, num_files = get_feature_iterator( manifest_path, sample_pct)
    #
    # iterator = generator()
    #
    # features_list = []
    # for features in tqdm.tqdm(iterator, total=num_files):
    #
    #     features_list.append(features)

    feature_reader = FeatureReader()
    features_list = FeatureReader.load_features(manifest_path, num_workers=num_workers)

    # Explicit clean up

    if flatten:
        return np.concatenate(features_list)

    return


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

    logger.info(f"Reading pre-computed w2vbert acoustic features...")

    features_batch = (
        read_features(
            args.manifest_path,
            num_workers=args.num_workers,
            sample_pct=args.sample_pct,
            flatten=True
        )
    )

    logger.info(f"Features shape = {features_batch.shape}\n")

    # Learn and save K-means model
    kmeans_model = get_kmeans_model(
        n_clusters=args.num_clusters,
        init=args.init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        tol=args.tol,
        max_no_improvement=args.max_no_improvement,
        n_init=args.n_init,
        reassignment_ratio=args.reassignment_ratio,
        random_state=args.seed,
    )
    logger.info("Starting k-means training...")
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