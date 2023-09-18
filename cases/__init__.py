"""Physical simulation case setups"""

from cases.cw import CW
from cases.db import DB
from cases.ldc import LDC
from cases.pf import PF
from cases.rlx import Rlx
from cases.rpf import RPF
from cases.tgv import TGV
from cases.ut import UTSetup


def select_case(case_name):
    """Select a simulation case to run"""
    cases = {
        "TGV": TGV,
        "RPF": RPF,
        "LDC": LDC,
        "PF": PF,
        "CW": CW,
        "DB": DB,
        "Rlx": Rlx,
        "UT": UTSetup,
    }
    return cases[case_name]
