
import numpy as np
from recursivenodes.metrics import (
        mass_matrix_condition,
        weak_laplacian_condition,
        strong_laplacian_condition,
        nodal_laplacian_condition,
        nodal_gradient_condition)
from recursivenodes import recursive_nodes
from recursivenodes.nodes import (
        warburton,
        blyth_luo_pozrikidis,
        rapetti_sommariva_vianello)


def _test_metric(d, n, nodes, metric, value, tol=1.e-3):
    v = metric(d, n, nodes)
    diff = np.abs(v - value)
    assert diff < value * tol


def test_metric():
    d = 2
    n = 4
    nodes = recursive_nodes(d, n, domain='biunit')
    _test_metric(d, n, nodes, mass_matrix_condition, 47.001337397112536)
    _test_metric(d, n, nodes, weak_laplacian_condition, 104.29722484760902)
    _test_metric(d, n, nodes, strong_laplacian_condition, 8.303572325075589)
    _test_metric(d, n, nodes, nodal_laplacian_condition, 8.1758189578178620)
    _test_metric(d, n, nodes, nodal_gradient_condition, 16.721533208965457)

    # Table 2 of Warburton (2006)
    vk_w = [
            5.9028,
            6.7769,
            7.8450,
            9.5913,
            11.1597,
            13.8858,
            16.8957,
            21.6675,
            27.4011,
            36.1156,
            47.1973,
            63.6592,
            85.6918,
            ]

    vk_bp = [
            5.9028,
            6.7763,
            7.7280,
            9.8423,
            11.4944,
            14.2101,
            18.0994,
            23.6271,
            31.4576,
            43.3978,
            61.0569,
            88.7706,
            130.2558,
            ]

    # These are not from any table: the condition numbers reported by RaSV12
    # for Warp & Blend nodes do not match the values reported by Warburton, so
    # I do not know how they were computed.  So these just check against
    # regression
    vk_rsv = [
            34.84352108770338,
            49.90049306800569,
            67.93576818216191,
            93.14306498697344,
            135.52857464463776,
            213.1424821831669,
            295.29190421689856,
            486.5966511834012,
            1008.1502912262075,
            582.5837145244341,
            965.4035836909193,
            2126.4281899860125,
            956.7186671454458,
            ]

    for (n, vkw, vkbp, vkrsv) in zip(range(3, 16), vk_w, vk_bp, vk_rsv):
        nw = warburton(d, n, domain='biunit')
        _test_metric(d, n, nw, mass_matrix_condition, vkw**2)
        nbp = blyth_luo_pozrikidis(d, n, domain='biunit')
        _test_metric(d, n, nbp, mass_matrix_condition, vkbp**2)
        nrsv = rapetti_sommariva_vianello(n, domain='biunit')
        _test_metric(d, n, nrsv, mass_matrix_condition, vkrsv)
