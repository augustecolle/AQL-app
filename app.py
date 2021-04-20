import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from scipy.stats import hypergeom


def get_pa(N, n, c, p):
    """
    Get the probability of acceptance
    N = lot size
    n = sample size
    c = tolerable defective rate
    p = defective rate
    """
    return hypergeom.cdf(c, N, N * p, n)


st.title("Incoming goods acceptance criteria")
lot_size = st.slider(
    "Lot size",
    min_value=int(1e3),
    max_value=int(1e5),
    value=int(1e4),
    step=int(1e3),
    format="%d",
)
acceptable_defective_rate = st.slider(
    "Maximum acceptable defective rate [%] @ rejection confidence",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
    format="%.1f",
)
rejection_confidence = st.slider(
    "Confidence of correct rejection [%]",
    min_value=50,
    max_value=100,
    value=75,
    step=1,
    format="%d",
)


def get_min_sample_size(
    lot_size, acceptable_defective_rate, rejection_confidence, _solution_tolerance=0.005
):
    solutions = []
    c_prev = 0
    for n in range(math.floor(lot_size / 200), int(0.05 * lot_size)):
        for c in range(1, math.ceil(acceptable_defective_rate * lot_size)):
            pa = get_pa(lot_size, n, c, acceptable_defective_rate)
            if (abs(pa - (1 - rejection_confidence)) <= _solution_tolerance) and (
                c_prev != c
            ):
                c_prev = c
                solutions.append((n, c))
    return solutions


res = get_min_sample_size(
    lot_size, acceptable_defective_rate / 100.0, rejection_confidence / 100.0
)

text = f"\\text{{A lot of {lot_size} units with a defective rate of {acceptable_defective_rate}\\,\\% will be rejected {rejection_confidence}\\,\\% of the time}}"
st.latex(text)

fig = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(
            text="Probability of acceptance in function of lot defective rate"
        )
    )
)
fig.update_xaxes(title_text="Defective rate of lot (p)")
fig.update_yaxes(title_text="Probability of acceptance")

for ri in res:
    p = np.linspace(0, acceptable_defective_rate / 100.0 + 0.01, 1000)
    pa = get_pa(lot_size, ri[0], ri[1], p)
    fig.add_trace(go.Scatter(x=p * 100, y=pa, name=f"({ri[0]}, {ri[1]})"))

st.plotly_chart(fig, use_container_width=True)
