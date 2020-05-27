""" log writer setup
"""

import sys

LogWriter = None

if int(sys.version[0]) == 3:
    from visualdl import LogWriter
    LogWriter = LogWriter

else:
    from tensorboardX import SummaryWriter
    LogWriter = SummaryWriter
