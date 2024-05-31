# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple
import os


def get_audio_files(manifest_path: str, extension: str) -> Tuple[str, List[str], List[int]]:
    fnames, sizes = [], []
    root_dir = ""
    with open(manifest_path, "r") as f:

        data = f.readlines()
        # root_dir = f.readline().strip()
        for line in data:
            items = line.strip().split()

            basename = os.path.basename(items[1]).replace(extension, "")
            fnames.append(basename)

    return root_dir, fnames, sizes

