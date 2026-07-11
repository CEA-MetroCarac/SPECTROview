"""Programmatic graph plotting mimicking the Graphs workspace."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, 
                 title: Optional[str] = None, ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled scatter plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, s=70, edgecolor='black')
    
    if title:
        ax.set_title(title)
        
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax

def plot_point(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, 
               join: bool = False, dodge: bool = True, title: Optional[str] = None, 
               ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled statistical point plot with error bars."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    sns.pointplot(data=df, x=x, y=y, hue=hue, errorbar=('ci', 95), 
                  join=join, dodge=dodge, capsize=0.1, ax=ax)
                  
    if title:
        ax.set_title(title)
        
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax

def plot_box(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, 
             title: Optional[str] = None, ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled box plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
    
    if title:
        ax.set_title(title)
        
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax

def plot_trendline(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, 
                   order: int = 1, title: Optional[str] = None, ax: Optional[plt.Axes] = None):
    """Create a SPECTROview-styled trendline (regression) plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    if hue is None:
        sns.regplot(data=df, x=x, y=y, order=order, ax=ax, 
                    scatter_kws={'s': 70, 'edgecolor': 'black'})
    else:
        sns.lmplot(data=df, x=x, y=y, hue=hue, order=order, 
                   scatter_kws={'s': 70, 'edgecolor': 'black'})
        # lmplot creates its own figure
        ax = plt.gca()
        
    if title:
        ax.set_title(title)
        
    ax.grid(True, linestyle='--', alpha=0.7)
    return ax
