# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from tabulate import tabulate
from IPython.display import HTML


def html_dict_table(values):
    return HTML(tabulate(values, headers=values.keys(), tablefmt="html"))
