# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BLEU metric implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import subprocess
import tempfile
import numpy as np
from os.path import isfile


from six.moves import urllib


def moses_multi_bleu(hypFileName, refFileName, lowercase=False):
  """Calculate the bleu score for hypotheses and references
  using the MOSES ulti-bleu.perl script.

  Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script

  Returns:
    The BLEU score as a float32 value.
  """

  # Get MOSES multi-bleu script
  
  eval_file = "/tmp/multi-bleu.perl"
  if not isfile(eval_file):
        multi_bleu_path, _ = urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
        "master/scripts/generic/multi-bleu.perl", eval_file)
        os.chmod(multi_bleu_path, 0o755)
        
  multi_bleu_path = eval_file

  # Calculate BLEU using multi-bleu script
  with open(hypFileName, "r") as read_pred:
    bleu_cmd = [multi_bleu_path]
    if lowercase:
      bleu_cmd += ["-lc"]
    bleu_cmd += [refFileName]
    
    try:
      bleu_out = subprocess.check_output(
          bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
      bleu_out = bleu_out.decode("utf-8")
      bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
      bleu_score = float(bleu_score)
    except subprocess.CalledProcessError as error:
      if error.output is not None:
        print("multi-bleu.perl script returned non-zero exit code")
        print(error.output)
      bleu_score = np.float32(0.0)

  # Close temp files
 


  return np.float32(bleu_score)
