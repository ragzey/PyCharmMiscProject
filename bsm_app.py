import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from options_pricing_engine import *


st.title('Black Scholes pricing model')

st.write('This is the interactive interface')


# Sidebar inputs
st.sidebar.header("Option variables")
s = st.sidebar.slider("Spot Price (S)", 50, 150, 100)
k = st.sidebar.slider("Strike Price (K)", 50, 150, 100)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.1, 0.05, step=0.01)
t = st.sidebar.slider("Time to Maturity (T in years)", 0.01, 2.0, 1.0)
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.2)

# Prices
call_price = call_bsm(s, k, r, t, sigma)
put_price = put_bsm(s, k, r, t, sigma)

st.subheader("Option Prices")
st.write(f"💰 Call Price: **{call_price:.4f}**")
st.write(f"💰 Put Price: **{put_price:.4f}**")

# Greeks
st.subheader("Option Greeks")
greeks = get_greeks(call_bsm, put_bsm, s, k, r, t, sigma)
st.json(greeks)

# Heatmaps
st.subheader("Volatility Heatmap")
heat_array_c, heat_array_p, s_list, vol_list = vol_heat_map(call_bsm, put_bsm, s, k, r, t, sigma)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].imshow(heat_array_c, cmap="viridis", extent=[vol_list[0], vol_list[-1], s_list[0], s_list[-1]], aspect='auto', origin='lower')
ax[0].set_title("Call Option Heatmap")
ax[0].set_xlabel("Volatility (σ)")
ax[0].set_ylabel("Spot Price (S)")

ax[1].imshow(heat_array_p, cmap="plasma", extent=[vol_list[0], vol_list[-1], s_list[0], s_list[-1]], aspect='auto', origin='lower')
ax[1].set_title("Put Option Heatmap")
ax[1].set_xlabel("Volatility (σ)")
ax[1].set_ylabel("Spot Price (S)")

st.pyplot(fig)