#!/usr/bin/env python
"""Evaluates sequence model.

This script assumes the gold and hypothesis data is stored in a two-column TSV
file, one example per line."""

__author__ = "Aaron Goyzueta, Kyle Gorman"

import argparse
import logging
import multiprocessing
import statistics

import evallib


def main(args: argparse.Namespace) -> None:
    wers = []
    lers = []
    for tsv_path in args.tsv_paths:
        # Word-level measures.
        correct = 0
        incorrect = 0
        # Label-level measures.
        total_edits = 0
        total_length = 0
        # Since the edit distance algorithm is quadratic, let's do this with
        # multiprocessing.
        with multiprocessing.Pool(args.cores) as pool:
            gen = pool.starmap(evallib.score, evallib.tsv_reader(tsv_path))
            for (edits, length) in gen:
                if edits == 0:
                    correct += 1
                else:
                    incorrect += 1
                total_edits += edits
                total_length += length
        wer = 100 * incorrect / (correct + incorrect)
        ler = 100 * total_edits / total_length
        wers.append(wer)
        lers.append(ler)
        print(f"{tsv_path}:\tWER:\t{wer:.2f}\tLER:\t{ler:.2f}")
    wer = statistics.mean(wers)
    ler = statistics.mean(lers)
    print(f"Macro-average:\tWER:\t{wer:.2f}\tLER:\t{ler:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluates sequence model")
    parser.add_argument(
        "tsv_paths", nargs="+", help="path to gold/hypo TSV file"
    )
    parser.add_argument(
        "--cores",
        default=multiprocessing.cpu_count(),
        help="number of cores (default: %(default)s)",
    )
    main(parser.parse_args())
