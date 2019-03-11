import argparse

from nmtg.models import get_model_type, get_model_names
from nmtg.preprocessors import get_preprocessor_names, get_preprocessor_type
from nmtg.trainers import get_trainer_names, get_trainer_type
from nmtg.tasks import get_task_names, get_task_type


class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        usage_text = super()._format_usage(
                usage, actions, groups, prefix)
        return '\n```\n{}\n```\n\n'.format(usage_text)

    def format_help(self):
        self._root_section.heading = '# %s' % self._prog
        return super().format_help()

    def start_section(self, heading):
        super().start_section('## **%s**' % heading)

    def _format_action(self, action):
        lines = []
        action_header = self._format_action_invocation(action)
        lines.append('### **%s** ' % action_header)
        if action.help:
            lines.append('')
            lines.append('```')
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
            lines.append('```')
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, **kwargs):
        super().__init__(
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=0,
                **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


def add_general_options(parser):
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')
    parser.add_argument('-seed', default=9999, type=int,
                        help='Seed for deterministic runs.')
    parser.add_argument('-cuda', action='store_true',
                        help='Use the GPU. Multi-GPU is not supported at this time')
    parser.add_argument('-fp16', action='store_true',
                        help='Use half precision training. Not compatible with all GPUs')


def add_task_option(parser):
    parser.add_argument('-task', default='translation', choices=get_task_names(),
                        help='Select which task to attempt')
    args, _ = parser.parse_known_args()

    task_class = get_task_type(args.task)
    return task_class


def add_trainer_option(parser):
    parser.add_argument('-trainer', default='nmt', choices=get_trainer_names(),
                        help='Select the trainer')
    args, _ = parser.parse_known_args()

    trainer_class = get_trainer_type(args.trainer)
    return trainer_class


def add_preprocessor_option(parser):
    parser.add_argument('preprocessor', default='bilingual', choices=get_preprocessor_names(),
                        help='Select the preprocessor')
    args, _ = parser.parse_known_args()

    preprocessor_class = get_preprocessor_type(args.preprocessor)
    return preprocessor_class
