from threading import Thread

import numpy as np


# Data


# Utils

def execute_on_thread(func: callable, args=()):
    thread = Thread(target=func, args=args)
    thread.start()
    return thread


def lerp(start, end, factor):
    return start + ((end - start) * factor)


def get_lerp_factor(start, end, value):
    return (value - start) / (end - start)


def map(value, start, end, new_start, new_end):
    return lerp(new_start, new_end, get_lerp_factor(start, end, value))


def is_index_valid(index: tuple, shape: tuple, allow_negative: bool = True):
    if len(index) > len(shape):
        return False

    for d in range(len(index)):
        i, l = index[d], shape[d]
        if allow_negative and i < 0:
            i += l

        if i < 0 or i >= l:
            return False
    return True


def get_neighbour_indices(pos: tuple, shape: tuple, upto_step: int = 1, diagonal: bool = True):
    row, col = pos
    res = []
    if upto_step <= 0:
        return res

    if diagonal:
        for s1 in range(max(0, row - upto_step), min(row + upto_step + 1, shape[0])):
            for s2 in range(max(0, col - upto_step), min(col + upto_step + 1, shape[1])):
                if s1 == row and s2 == col:
                    continue

                _pos = (s1, s2)
                if is_index_valid(_pos, shape, allow_negative=False):
                    res.append(_pos)
    else:
        for s in range(1, upto_step + 1):
            for _pos in ((row - s, col),
                         (row, col + s),
                         (row + s, col),
                         (row, col - s)):
                if is_index_valid(_pos, shape, allow_negative=False):
                    res.append(_pos)

    return res
