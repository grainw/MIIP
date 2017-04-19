# coding=utf-8


class lo:
    def __init__(self):
        pass

    @staticmethod
    def sub_bool(lst,idx):
        return [itm for i,itm in enumerate(lst) if idx[i]]

    @staticmethod
    def sub_int(lst,idx):
        return [itm for i, itm in enumerate(lst) if i in idx]

    @staticmethod
    def valid_idx(lst):
        return [i for i, x in enumerate(lst) if x.strip()]

    @staticmethod
    def empty_idx(lst):
        return [i for i, x in enumerate(lst) if not x.strip()]