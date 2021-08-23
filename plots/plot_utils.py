import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import collections  as mc

lg = '#808080'

def clean_axis(ax):
   ax.spines["top"].set_visible(False)
   ax.spines["right"].set_visible(False)
   ax.spines['bottom'].set_color(lg)
   ax.spines['left'].set_color(lg)
   ax.tick_params(axis='x', colors=lg)
   ax.tick_params(axis='y', colors=lg)
   ax.set_ylim(0.5, 1)
