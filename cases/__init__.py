"""Physical simulation case setups"""

from cases.cw import CW
from cases.db import DB
from cases.ht import HT
from cases.ldc import LDC
from cases.pf import PF
from cases.rpf import RPF
from cases.tgv import TGV
from cases.ut import UTSetup


def select_case(case_name):
    """Select a simulation case to run"""
    cases = {
        "tgv": TGV,
        "rpf": RPF,
        "ldc": LDC,
        "pf": PF,
        "cw": CW,
        "db": DB,
        "ut": UTSetup,
        "ht": HT,
    }
    return cases[case_name]
