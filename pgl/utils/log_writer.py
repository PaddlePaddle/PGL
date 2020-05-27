""" log writer setup
"""

import sys

log_writer = None

if sys.version[0] == 3:
    from visualdl import LogWriter
    log_writer = LogWriter

else:
    from tensorboardX import SummaryWriter
    log_writer = SummaryWriter
