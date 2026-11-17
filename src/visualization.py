from typing import List

import plotly.express as px
import pandas as pd


def plot_value_counts(series: pd.Series, title: str = 'Distribution'):
    counts = series.value_counts().reset_index()
    counts.columns = ['class', 'count']
    return px.bar(counts, x='class', y='count', title=title, color='class', template='plotly_dark')


def plot_feature_importances(model, feature_names: List[str], top_n: int = 10):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        frame = pd.DataFrame({'feature': feature_names, 'importance': importances})
        frame = frame.sort_values('importance', ascending=False).head(top_n)
        fig = px.bar(frame, x='importance', y='feature', orientation='h', title='Top Feature Importances', template='plotly_dark')
        return fig
    return None


def plot_dataset_overview(frame: pd.DataFrame):
    counts = frame['Label'].value_counts().reset_index()
    counts.columns = ['Label', 'count']
    return px.bar(counts, x='Label', y='count', title='Traffic Distribution', template='plotly_dark')
