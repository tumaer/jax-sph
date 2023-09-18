"""Unit Test simulator"""


def UTSimulator(g_ext_fn):
    """Unit Test simulator: apply gravity as only force"""

    def forward(state, neighbors):
        g_ext = g_ext_fn(state["r"])

        state["dudt"] = g_ext
        return state

    return forward
