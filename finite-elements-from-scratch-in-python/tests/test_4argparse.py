import pytest
import numpy as np

pytest.importorskip("scipy")

from recursivenodes.lebesgue import (
        add_lebesguemax_to_parser,
        lebesguemax_from_args)


def test_parse_and_measure(argv, tol=1.e-3):
    from argparse import ArgumentParser

    max_l_target = argv[-1]
    argv = argv[:-1]
    parser = ArgumentParser(
            description='Test ArgumentParser interface to lebesguemax'
            )
    add_lebesguemax_to_parser(parser)
    args = parser.parse_args(args=argv)
    (max_l, max_x) = lebesguemax_from_args(args)
    assert np.abs(max_l - max_l_target) < max_l_target * tol
