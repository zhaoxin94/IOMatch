from .fixmatch import FixMatch
from .openmatch import OpenMatch
from .iomatch import IOMatch
from .labelonly import LabelOnly
from .scomatch import ScoMatch
from .nsmatch import NSMatch
from .psmatch import PSMatch
from .psmatch2 import PSMatch2

# if any new alg., please append the dict
name2alg = {
    "fixmatch": FixMatch,
    'openmatch': OpenMatch,
    "iomatch": IOMatch,
    "labelonly": LabelOnly,
    "scomatch": ScoMatch,
    "nsmatch": NSMatch,
    "psmatch": PSMatch,
    "psmatch2": PSMatch2
}


def get_algorithm(args, net_builder, tb_log, logger):
    try:
        alg = name2alg[args.algorithm](args=args,
                                       net_builder=net_builder,
                                       tb_log=tb_log,
                                       logger=logger)
        return alg
    except KeyError as e:
        print(f'Unknown algorithm: {str(e)}')
