#!/usr/bin/env python
"""Evaluation functions for sequence models."""

__author__ = "Kyle Gorman"

import logging

import numpy  # type: ignore

from typing import Any, Iterator, List, Tuple


Labels = List[Any]


def edit_distance(x: Labels, y: Labels) -> int:
    # For a more expressive version of the same, see:
    #
    #     https://gist.github.com/kylebgorman/8034009
    idim = len(x) + 1
    jdim = len(y) + 1
    table = numpy.zeros((idim, jdim), dtype=numpy.uint8)
    table[1:, 0] = 1
    table[0, 1:] = 1
    for i in range(1, idim):
        for j in range(1, jdim):
            if x[i - 1] == y[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                c1 = table[i - 1][j]
                c2 = table[i][j - 1]
                c3 = table[i - 1][j - 1]
                table[i][j] = min(c1, c2, c3) + 1
    return int(table[-1][-1])


def score(gold: Labels, hypo: Labels) -> Tuple[int, int]:
    """Computes sufficient statistics for LER calculation."""
    edits = edit_distance(gold, hypo)
    if edits:
        logging.warning(
            "Incorrect prediction:\t%r (predicted: %r)",
            " ".join(gold),
            " ".join(hypo),
        )
    return (edits, len(gold))


def tsv_reader(path: str) -> Iterator[Tuple[Labels, Labels]]:
    """Reads pairs of strings from a TSV filepath."""
    with open(path, "r") as source:
        for line in source:
            (gold, hypo) = line.split("\t", 1)
            # Stripping is performed after the fact so the previous line
            # doesn't fail when `hypo` is null.
            hypo = hypo.rstrip()
            yield (gold.split(), hypo.split())
