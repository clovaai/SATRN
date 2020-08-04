"""
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import os
import yaml
import collections


def dict_to_namedtuple(d):
    """ Convert dictionary to named tuple.
    """
    FLAGSTuple = collections.namedtuple('FLAGS', sorted(d.keys()))

    for k, v in d.items():

        if type(v) is dict:
            d[k] = dict_to_namedtuple(v)

        elif type(v) is str:
            try:
                d[k] = eval(v)
            except:
                d[k] = v

    nt = FLAGSTuple(**d)

    return nt


class Flags:
    """ Flags object.
    """

    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            d = yaml.safe_load(f)

        self.flags = dict_to_namedtuple(d)

    def get(self):
        return self.flags
