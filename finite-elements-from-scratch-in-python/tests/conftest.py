def pytest_configure(config):
    import numpy as np
    np.random.seed(1)


def pytest_generate_tests(metafunc):
    if "d" in metafunc.fixturenames:
        metafunc.parametrize("d", range(1, 4))
    if "k" in metafunc.fixturenames:
        metafunc.parametrize("k", range(3, 7))
    if "d_ref" in metafunc.fixturenames:
        metafunc.parametrize("d_ref", range(2, 4))
    if "k_ref" in metafunc.fixturenames:
        metafunc.parametrize("k_ref", range(4, 8))
    if "C" in metafunc.fixturenames:
        metafunc.parametrize("C", [False, True])
    if "nodes" in metafunc.fixturenames:
        metafunc.parametrize("nodes", [
            'equispaced',
            'blyth_luo_pozrikidis',
            'warburton',
            'recursive',
            ])
    if "argv" in metafunc.fixturenames:
        metafunc.parametrize("argv", [
            tuple(
                '-d 2 -n 4 -v --nodes equispaced_interior'.split()
                ) + (15.738692897773845,),
            tuple(
                '-d 2 -n 4 -v --family lgc'.split()
                ) + (2.8496378329485137,),
            tuple(
                '-d 2 -n 4 -v --family lgg --g-alpha 0.25'.split()
                ) + (2.7442003656245304,),
            tuple(
                '-d 2 -n 4 -v --family gl'.split()
                ) + (5.1102267257704925,),
            tuple(
                '-d 2 -n 4 -v --family gc'.split()
                ) + (3.26225564737967,),
            tuple(
                '-d 2 -n 4 -v --family gg --g-alpha 0.25'.split()
                ) + (4.121522189061396,),
            tuple(
                '-d 2 -n 4 -v --family equi_interior --domain equilateral'.split()
                ) + (19.67886583176972,),
            tuple(
                '-d 2 -n 4 -v --nodes equispaced --domain barycentric'.split()
                ) + (3.4748158628553973,),
            tuple(
                '-d 2 -n 4 -v --nodes blyth_luo_pozrikidis'.split()
                ) + (2.662095048971729,),
            tuple(
                '-d 2 -n 4 -v --nodes warburton'.split()
                ) + (2.6622188557840967,),
            ])
