import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts


def getBoxplot(data: list, x_ticks: list, x_label: str, y_label: str, xlabel_rot: int=85, figSize: tuple=(15,8)):
    sns.set(rc = {'figure.figsize':(15,8)})

    ax = sns.boxplot(data=data)
    ax.set_xticklabels(x_ticks, rotation=xlabel_rot)
    ax.set(xlabel=x_label, ylabel=y_label)
    return ax


def getErrorplot(xaxis:np.ndarray, yaxis: list, errors: np.ndarray, x_ticks: list, title: str, x_label: str, format: str = 'o', color: str='k'):
    plt.figure()
    plt.errorbar(xaxis, yaxis, xerr=errors, fmt = format, color = color)
    plt.yticks(yaxis, x_ticks)
    plt.title(title)
    plt.xlabel(x_label)


def getKMplot(idx: list, os_time: list, cnsr_events: list, title: str):
    ax = plt.subplot(111)

    ax.set_title('KM-curve for Predicted High vs Low risk groups')

    kmf_low = KaplanMeierFitter()
    ax = kmf_low.fit(os_time[idx], cnsr_events[idx], label='Pred_Low').plot_survival_function(ax=ax)

    kmf_high = KaplanMeierFitter()
    ax = kmf_high.fit(os_time[~idx], cnsr_events[~idx], label='Pred_high').plot_survival_function(ax=ax)

    add_at_risk_counts(kmf_high, kmf_low, ax=ax)
    plt.tight_layout()
    return ax
