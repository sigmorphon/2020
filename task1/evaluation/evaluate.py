#!/usr/bin/env python
"""Evaluates sequence model.

This script assumes the gold and hypothesis data is stored in a two-column TSV
file, one example per line."""

__author__ = "Kyle Gorman"

import argparse
import logging
import multiprocessing

import numpy

from typing import Any, Iterator, List, Tuple


Labels = List[Any]


def _edit_distance(x: Labels, y: Labels) -> int:
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


def _score(gold: Labels, hypo: Labels) -> Tuple[int, int]:
    """Computes sufficient statistics for LER calculation."""
    edits = _edit_distance(gold, hypo)
    if edits:
        logging.warning(
            "Incorrect prediction:\t%r (predicted: %r)",
            " ".join(gold),
            " ".join(hypo),
        )
    return (edits, len(gold))


def _tsv_reader(path: str) -> Iterator[Tuple[Labels, Labels]]:
    """Reads pairs of strings from a TSV filepath."""
    with open(path, "r") as source:
        for line in source:
            (gold, hypo) = line.split("\t", 1)
            # Stripping is performed after the fact so the previous line
            # doesn't fail when `hypo` is null.
            hypo = hypo.rstrip()
            yield (gold.split(), hypo.split())


def main(args: argparse.Namespace) -> None:
    # Word-level measures.
    correct = 0
    incorrect = 0
    # Label-level measures.
    total_edits = 0
    total_length = 0
    # Since the edit distance algorithm is quadratic, let's do this with
    # multiprocessing.
    with multiprocessing.Pool(args.cores) as pool:
        gen = pool.starmap(_score, _tsv_reader(args.tsv_path))
        for (edits, length) in gen:
            if edits == 0:
                correct += 1
            else:
                incorrect += 1
            total_edits += edits
            total_length += length
    print(f"WER:\t{100 * incorrect / (correct + incorrect):.2f}")
    print(f"LER:\t{100 * total_edits / total_length:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluates sequence model")
    parser.add_argument("tsv_path", help="path to gold/hypo TSV file")
    parser.add_argument(
        "--cores",
        default=multiprocessing.cpu_count(),
        help="number of cores (default: %(default)s)",
    )
    main(parser.parse_args())
