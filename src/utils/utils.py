import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import norm
import math


def to_snake_case(name: str) -> str:
    """
    Convierte nombres de columnas a snake_case.
    Ejemplos:
    'TotalCharges' -> 'total_charges'
    'Monthly Charges' -> 'monthly_charges'
    """
    name = name.strip()
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)   # separa CamelCase
    name = re.sub(r"[^\w]+", "_", name)            # sustituye espacios y símbolos
    name = re.sub(r"_+", "_", name).strip("_")     # limpia guiones bajos repetidos
    return name.lower()



def resumen_nulos_ceros(df):
    """
    Genera un Data Frame con el tipo de dato, la cantidad y porcentaje 
    de valores nulos y la cantidad y porcentaje de ceros para cada varaible
    
    Parámetros:
    df: pandas.DataFrame
        El dataframe que se desea analizar
    
    Output:
    pandas.io.formats.style.Styler
        Un objeto DataFrame estilizado con un gradiente de color para
        nulos y ceros
    """

    # Creamos el dataframe descriptivo base

    desc_df = pd.DataFrame({
        'variable': df.columns,
        'variable_type': df.dtypes.astype(str).values,
        'n_missing': df.isna().sum().values,
        'n_zeros': (df==0).sum().values,
        'complete_rate': 1-(df.isna().sum().values/len(df))
    })

    # Manipulacuón mediante Methon Chaining (estilo dplyr)
    var_type_missing_df = (
        desc_df
        .assign(
            n_missing_perc = lambda x: (100*(1-x['complete_rate'])).round(3),
            n_zeros_perc = lambda x: (100*(x['n_zeros']/len(df))).round(3)
        )
        .filter(['variable_type', 'variable', 'n_missing', 'n_missing_perc', 'n_zeros', 'n_zeros_perc'])
        .sort_values(['variable_type', 'n_missing'])
    )

    return (var_type_missing_df.style
            .background_gradient(subset = ['n_missing_perc'], cmap = 'Reds')
            .background_gradient(subset = ['n_zeros_perc'], cmap = 'Blues'))



def histogram_plot(data, var, ax = None, bins = 50):
    
    if var not in data.columns:
        raise ValueError("La variable especificada no existe en el dataframe")
    
    data_na_omit = data[var].dropna()
    
    media = data_na_omit.mean()
    desv = data_na_omit.std()
    
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
    
    sns.histplot(data_na_omit, bins = bins, stat = "density", ax = ax)
    
    x = np.linspace(data_na_omit.min(), data_na_omit.max(), 1000)
    ax.plot(x, norm.pdf(x, media, desv), color = "red")
    
    ax.set_title(f"Histograma de {var}")
    ax.set_ylabel("Densidad")


def histogram_var_target_plot(data, var, target_var = "Class", ax = None, bins = 50):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    if var not in data.columns or target_var not in data.columns:
        raise ValueError("Una o ambas variables especificadas no existen en el dataframe")
    
    data_na_omit = data[[var, target_var]].dropna(subset = [var])
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.histplot(
        data = data_na_omit,
        x = var,
        hue = target_var,
        bins = bins,
        stat = "density",
        common_norm = False,
        alpha = 0.5,
        ax = ax
    )
    
    ax.set_title(f"Histograma de {var} según {target_var}")
    ax.set_ylabel("Densidad")


def corrplot(data):
    """
    Calcula matriz de correlaciones y visualiza en un heatmap sin números
    """
    correlacion_pearson_df = data.corr(method='pearson')

    # Visualizar la matriz de correlación
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.2)
    
    # Cambiamos annot a False
    sns.heatmap(correlacion_pearson_df, annot=False, cmap='coolwarm')
    
    plt.title('Matriz de Correlación de Pearson')
    plt.show()
