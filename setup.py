from setuptools import setup, find_packages

setup(name='onmt',
      version='0.1',
      author='quanpn90',
      author_email='ngoc.pham@kit.edu',
      url='https://github.com/quanpn90/NMTGMinor',
      license='MIT',
      scripts=[
          'train.py', 'translate.py', 'autoencoder.py',
          'average_checkpoints_auto.py', 'eval_autoencoder.py',
          'grad_check.py', 'grad_check_relative_attention.py', 'online.py',
          'preprocess_multi_dataset.py', 'preprocess.py',
          'rematch_language_embedding.py', 'rescore.py', 'sample_lm.py',
          'test_reversible.py', 'train_distributed.py',
          'train_language_model.py', 'train.py', 'translate_distributed.py',
          'translate.py'
      ],
      packages=find_packages(),
      install_requires=['pytorch', 'apex', 'h5py'])
