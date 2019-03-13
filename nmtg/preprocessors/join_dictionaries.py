import os

from nmtg.data import Dictionary
from nmtg.preprocessors import register_preprocessor, Preprocessor


@register_preprocessor('join_dicts')
class JoinDictionaries(Preprocessor):
    @classmethod
    def add_options(cls, parser):
        super().add_options(parser)
        parser.add_argument('dicts', nargs='+')
        parser.add_argument('-out_name', default='dict')

    @classmethod
    def preprocess(cls, args):
        dictionaries = [Dictionary.load(filename) for filename in args.dicts]
        dictionary = dictionaries[0]
        for x in dictionaries[1:]:
            dictionary.update(x)

        dictionary.save(os.path.join(args.data_dir_out, args.out_name))
