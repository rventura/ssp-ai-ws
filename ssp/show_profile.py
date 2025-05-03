#!/usr/bin/env python3

import sys
from pstats import *

p = Stats(sys.argv[1] if len(sys.argv)>1 else "out.profile")
p.sort_stats(SortKey.CUMULATIVE).print_stats(100)
