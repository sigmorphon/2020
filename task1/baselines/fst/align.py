#!/usr/bin/env python
"""Aligner for grapheme-to-phoneme model training.

The input is a two-column TSV file (with no escapes) where the first column
consists of the graphemic form and the second consists of the corresponding
phonemic transcription.

The output consist of a FAR (FST archives) of alignments, encoded as unweighted
FSAs, and the corresponding encoder table.

Synopsis
--------

In stage one (_lexicon_covering), we build FARs that contain the grapheme and
phoneme strings, respectively, and also build a zeroth order Markov model
covering grammar FST.

In stage two (_alignments), we set the covering grammar probabilities using
expectation maximization, then decodes the training corpus using this model.
This requires the command-line tools `baumwelchtrain` and `baumwelchdecode`
available here:

    http://www.openfst.org/twiki/pub/Contrib/FstContrib/baumwelch-0.3.0.tar.gz

In stage three (_encode), we finally encode the alignment FSTs as FSAs. This
allows us to construct n-gram models over the pairs using the OpenGrm-NGram
command line tools available here:

    http://ngram.opengrm.org

The encoder table is also written out so the resulting model can be decoded,
producing a final WFSA.

Nota bene
---------

This makes use of several temporary files. Temporary files are created using
`tempfile.mkstemp`. If you wish to generate these files in a different
directory than the OS default, set the $TMPDIR, $TEMP, or $TMP environmental
variables.

All temporary files are removed when the PairNGramAligner object is deleted.

To see shell commands as they are invoked, set the log level to DEBUG."""

__author__ = "Kyle Gorman"


import argparse
import functools
import logging
import multiprocessing
import shutil
import subprocess
import tempfile
import operator
import os
import random
import re
import time

from typing import Set, Tuple, Union

import pynini
import pywrapfst


TokenType = Union[str, pynini.SymbolTable]

TOKEN_TYPES = ["byte", "utf8"]
DEV_NULL = open(os.devnull, "w")
INF = float("inf")
RAND_MAX = 32767


def _str_to_bool(value: str) -> bool:
    """Handler for string-like boolean flag types."""
    value = value.lower()
    if value in ("true", "1"):
        return True
    elif value in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected; got {value}")


class PairNGramAligner:
    """Produces FSA alignments for pair n-gram model training."""

    _compactor = functools.partial(
        pywrapfst.convert, fst_type="compact_string"
    )

    def __init__(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.g_path = os.path.join(self.tempdir.name, "g.far")
        self.p_path = os.path.join(self.tempdir.name, "p.far")
        self.c_path = os.path.join(self.tempdir.name, "c.fst")
        self.align_path = os.path.join(self.tempdir.name, "align.fst")
        self.afst_path = os.path.join(self.tempdir.name, "afst.far")

    def __del__(self):
        self.tempdir.cleanup()

    def align(
        self,
        # Input TSV path.
        tsv_path: str,
        # Output FAR path.
        far_path: str,
        encoder_path: str,
        # Arguments for constructing the lexicon and covering grammar.
        input_token_type: TokenType,
        input_epsilon: bool,
        output_token_type: TokenType,
        output_epsilon: bool,
        # Arguments used during the alignment phase.
        cores: int,
        random_starts: int,
        seed: int,
        delta: str = "",
        fst_default_cache_gc: str = "",
        fst_default_cache_gc_limit: str = "",
        max_iters: str = "",
    ):
        """Runs the entire alignment regimen."""
        self._lexicon_covering(
            tsv_path,
            input_token_type,
            input_epsilon,
            output_token_type,
            output_epsilon,
        )
        self._alignments(
            cores,
            random_starts,
            seed,
            delta,
            fst_default_cache_gc,
            fst_default_cache_gc_limit,
            max_iters,
        )
        self._encode(far_path, encoder_path)
        logging.info(
            "Success! FAR path: %s; encoder path: %s", far_path, encoder_path
        )

    @staticmethod
    def _label_union(labels: Set[int], epsilon: bool) -> pynini.Fst:
        """Creates FSA over a union of the labels."""
        side = pynini.Fst()
        src = side.add_state()
        side.set_start(src)
        dst = side.add_state()
        if epsilon:
            labels.add(0)
        one = pynini.Weight.One(side.weight_type())
        for label in labels:
            side.add_arc(src, pynini.Arc(label, label, one, dst))
        side.set_final(dst)
        assert side.verify(), "FST is ill-formed"
        return side

    @staticmethod
    def _narcs(f: pynini.Fst) -> int:
        """Computes the number of arcs in an FST."""
        return sum(f.num_arcs(state) for state in f.states())

    NON_SYMBOL = ("byte", "utf8")

    def _lexicon_covering(
        self,
        tsv_path: str,
        input_token_type: TokenType,
        input_epsilon: bool,
        output_token_type: TokenType,
        output_epsilon: bool,
    ) -> None:
        """Builds covering grammar and lexicon FARs."""
        # Sets of labels for the covering grammar.
        g_labels: Set[int] = set()
        p_labels: Set[int] = set()
        # Curries compiler functions for the FARs.
        icompiler = functools.partial(
            pynini.acceptor, token_type=input_token_type
        )
        ocompiler = functools.partial(
            pynini.acceptor, token_type=output_token_type
        )
        logging.info("Constructing grapheme and phoneme FARs")
        g_writer = pywrapfst.FarWriter.create(self.g_path)
        p_writer = pywrapfst.FarWriter.create(self.p_path)
        with open(tsv_path, "r") as source:
            for (linenum, line) in enumerate(source, 1):
                key = f"{linenum:08x}"
                (g, p) = line.rstrip().split("\t", 1)
                # For both G and P, we compile a FSA, store the labels, and
                # then write the compact version to the FAR.
                g_fst = icompiler(g)
                g_labels.update(g_fst.paths().ilabels())
                g_writer[key] = self._compactor(g_fst)
                p_fst = ocompiler(p)
                p_labels.update(p_fst.paths().ilabels())
                p_writer[key] = self._compactor(p_fst)
        logging.info("Processed %s examples", f"{linenum:,d}")
        logging.info("Constructing covering grammar")
        logging.info("%d unique graphemes", len(g_labels))
        g_side = self._label_union(g_labels, input_epsilon)
        logging.info("%d unique phones", len(p_labels))
        p_side = self._label_union(p_labels, output_epsilon)
        # The covering grammar is given by (G x P)^*.
        covering = pynini.transducer(g_side, p_side).closure().optimize()
        assert covering.num_states() == 1, "Covering grammar FST is ill-formed"
        logging.info(
            "Covering grammar has %s arcs",
            f"{PairNGramAligner._narcs(covering):,d}",
        )
        covering.write(self.c_path)

    @staticmethod
    def _random_start(*args: str) -> Tuple[str, float]:
        """Performs a single random start."""
        (*cmd, idx) = args
        start = time.time()
        likelihood = INF
        logging.debug("Subprocess call: %s", cmd)
        with subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True) as proc:
            # Parses STDERR to capture the likelihood.
            for line in proc.stderr:
                match = re.match(
                    r"INFO: Best likelihood:\s(-?\d*(\.\d*))", line
                )
                if match:
                    likelihood = float(match.group(1))
                    logging.info(
                        "Random start %d; likelihood: %f; time elapsed: %ds",
                        idx,
                        likelihood,
                        time.time() - start,
                    )
        return (cmd[-1], likelihood)

    def _alignments(
        self,
        cores: int,
        random_starts: int,
        seed: int,
        delta: str = "",
        fst_default_cache_gc: str = "",
        fst_default_cache_gc_limit: str = "",
        max_iters: str = "",
    ) -> None:
        """Trains the aligner and constructs the alignments FAR."""
        logging.info("Training aligner")
        cmd_fixed = ["baumwelchtrain", "--expectation_table=ilabel"]
        if delta:
            cmd_fixed.append(f"--delta={delta}")
        if fst_default_cache_gc:
            cmd_fixed.append(f"--fst_default_cache_gc={fst_default_cache_gc}")
        if fst_default_cache_gc_limit:
            cmd_fixed.append(
                f"--fst_default_cache_gc_limit={fst_default_cache_gc_limit}"
            )
        if max_iters:
            cmd_fixed.append(f"--max_iters={max_iters}")
        # Adds more arguments shared across all commands.
        if max_iters:
            cmd_fixed.append(f"--max_iters={max_iters}")
        cmd_fixed.append("--remove_zero_arcs=false")
        cmd_fixed.append("--flat_start=false")
        cmd_fixed.append("--random_starts=1")
        # Constructs the actual command vectors (plus an index for logging
        # purposes).
        random.seed(seed)
        commands = [
            (
                *cmd_fixed,
                f"--seed={seed}",
                self.g_path,
                self.p_path,
                self.c_path,
                os.path.join(self.tempdir.name, f"{seed:010d}.fst"),
                idx,
            )
            for (idx, seed) in enumerate(
                random.sample(range(1, RAND_MAX), random_starts), 1
            )
        ]
        # Actually runs starts.
        logging.info("Random starts")
        with multiprocessing.Pool(cores) as pool:
            # Setting chunksize to 1 means that random starts are processed
            # in roughly the order you'd expect.
            gen = pool.starmap(self._random_start, commands, chunksize=1)
            # Because we're in negative log space.
            (best_fst, best_likelihood) = min(gen, key=operator.itemgetter(1))
        logging.info("Best likelihood: %f", best_likelihood)
        # Moves best likelihood solution to the requested location.
        shutil.move(best_fst, self.align_path)
        logging.info("Computing alignments")
        cmd = ["baumwelchdecode"]
        if fst_default_cache_gc:
            cmd.append(f"--fst_default_cache_gc={fst_default_cache_gc}")
        if fst_default_cache_gc_limit:
            cmd.append(
                f"--fst_default_cache_gc_limit={fst_default_cache_gc_limit}"
            )
        cmd.append(self.g_path)
        cmd.append(self.p_path)
        cmd.append(self.align_path)
        cmd.append(self.afst_path)
        logging.debug("Subprocess call: %s", cmd)
        subprocess.check_call(cmd)

    def _encode(self, far_path: str, encoder_path: str) -> None:
        """Encodes the alignments."""
        logging.info("Encoding the alignments as FSAs")
        encoder = pywrapfst.EncodeMapper(encode_labels=True)
        a_reader = pywrapfst.FarReader.open(self.afst_path)
        a_writer = pywrapfst.FarWriter.create(far_path)
        # Curries converter function for the FAR.
        converter = functools.partial(pywrapfst.convert, fst_type="vector")
        while not a_reader.done():
            key = a_reader.get_key()
            fst = converter(a_reader.get_fst())
            fst.encode(encoder)
            a_writer[key] = self._compactor(fst)
            a_reader.next()
        encoder.write(encoder_path)


def main(args: argparse.Namespace) -> None:
    aligner = PairNGramAligner()
    input_token_type = (
        args.input_token_type
        if args.input_token_type in TOKEN_TYPES
        else pynini.SymbolTable.read_text(args.input_token_type)
    )
    output_token_type = (
        args.output_token_type
        if args.output_token_type in TOKEN_TYPES
        else pynini.SymbolTable.read_text(args.output_token_type)
    )
    aligner.align(
        args.tsv_path,
        args.far_path,
        args.encoder_path,
        input_token_type,
        args.input_epsilon,
        output_token_type,
        args.output_epsilon,
        args.cores,
        args.random_starts,
        args.seed,
        args.delta,
        args.fst_default_cache_gc,
        args.fst_default_cache_gc_limit,
        args.max_iters,
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Aligner for grapheme-to-phoneme model training"
    )
    # Arguments for constructing the lexicon and covering grammar.
    parser.add_argument(
        "--tsv_path", required=True, help="input TSV file path"
    )
    parser.add_argument("--far_path", required=True, help="output FAR path")
    parser.add_argument(
        "--encoder_path", required=True, help="output encoder path"
    )
    parser.add_argument(
        "--input_token_type",
        default="utf8",
        help="input token type (default: %(default)s)",
    )
    parser.add_argument(
        "--input_epsilon",
        type=_str_to_bool,
        default=True,
        help="allows graphemes to have null alignments (default: %(default)s)",
    )
    parser.add_argument(
        "--output_token_type",
        default="utf8",
        help="output token type (default: %(default)s)",
    )
    parser.add_argument(
        "--output_epsilon",
        type=_str_to_bool,
        default=True,
        help="allows phonemes to have null alignments (default: %(default)s)",
    )
    # Arguments used during the alignment phase.
    parser.add_argument(
        "--cores",
        type=int,
        default=multiprocessing.cpu_count(),
        help="number of cores (default: %(default)s)",
    )
    parser.add_argument(
        "--random_starts",
        type=int,
        default=10,
        help="number of random starts (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    # Unless otherwise marked, all of the following arguments are passed to
    # lower-level binaries as strings, use `true` as a value to get the normal
    # boolean interpretation.
    parser.add_argument("--delta")
    parser.add_argument("--fst_default_cache_gc")
    parser.add_argument("--fst_default_cache_gc_limit")
    parser.add_argument("--max_iters")
    main(parser.parse_args())
