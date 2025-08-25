#!/usr/bin/env python3

import sys
import lxml.etree as etree

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        f = open(fname, 'r')
    else:
        f = sys.stdin

    try:
        x = etree.parse(f, None)
        sys.stdout.write(etree.tostring(x, pretty_print=True, encoding=str))  # type: ignore
    except KeyboardInterrupt:
        pass
