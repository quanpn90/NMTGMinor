from setuptools import setup

setup(
    name='NMTGMinor',
    version='1.0',
    description='A Neural Machine Translation toolkit for research purpose',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/quanpn90/NMTGMinor',
    author='Quan-Pham Ngoc, Felix Schneider',
    packages=['nmtg'],
    scripts=['train.py', 'score_results.py', 'preprocess.py', 'online_translation.py', 'get_validation_loss.py',
             'evaluate.py', 'average_checkpoints.py',
             'tools/generate_vocabulary.py', 'tools/language_noise.py', 'tools/validate.py']
)
