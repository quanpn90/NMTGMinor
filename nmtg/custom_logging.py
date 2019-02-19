import logging
import tqdm
import time
import os
import sys


class TQDMLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    # noinspection PyBroadException
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


LOG_LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}


def setup_logging(log_dir, filename, file_level, console_level):
    handlers = [TQDMLoggingHandler(level=LOG_LEVELS[console_level])]

    if log_dir is not None:
        if not os.path.isdir(log_dir):
            if os.path.exists(log_dir):
                raise ValueError('Log directory exists and is a file')
            os.makedirs(log_dir)
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        filename = os.path.join(log_dir, '{} {}.log'.format(t, filename))
        handler = logging.FileHandler(filename, 'w')
        handler.setLevel(LOG_LEVELS[file_level])
        handlers.append(handler)

    logging.basicConfig(
        datefmt='%Y-%m-%d %H:%M:%S',
        format='{asctime} | {levelname}: {message}',
        style='{',
        level=min(LOG_LEVELS[file_level], LOG_LEVELS[console_level]),
        handlers=handlers
    )

    logger = logging.getLogger()

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return logger


def setup_logging_from_args(args, filename):
    return setup_logging(args.log_dir, filename, args.log_level_file, args.log_level_console)


def add_log_options(parser):
    parser.add_argument('-log_dir', type=str,
                        help='Where to place log files')
    parser.add_argument('-log_level_file', type=str, default='debug',
                        help='Logging level for the file logger. '
                             'Accepts debug (default), info, warning, error, critical')
    parser.add_argument('-log_level_console', type=str, default='debug',
                        help='Logging level for the console logger. '
                             'Accepts debug (default), info, warning, error, critical')
    parser.add_argument('-no_progress', action='store_true',
                        help='Disable progress bars')
