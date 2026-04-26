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
    name = name.strip() # Elimina espacios invisibles al inicio o al final
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)   # separa CamelCase
    name = re.sub(r"[^\w]+", "_", name)            # sustituye espacios y símbolos
    name = re.sub(r"_+", "_", name).strip("_")     # limpia guiones bajos repetidos
    return name.lower()




def resumen_nulos_ceros(df):
    """
    Genera un Data Frame estilizado con el tipo de dato, cantidad y porcentaje 
    de valores nulos y ceros, compatible con cualquier tipo de dato.
    """

    # Calculamos nulos 
    n_missing = df.isna().sum()

    # Calculamos ceros 
    # Solo intentamos comparar con 0 en columnas numéricas para evitar errores
    # y excluimos booleanos para que False no cuente como 0.
    def contar_ceros(col):
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            return (col == 0).sum()
        else:
            return 0

    n_zeros = df.apply(contar_ceros)

    # Creamos el dataframe descriptivo base
    desc_df = pd.DataFrame({
        'variable': df.columns,
        'variable_type': df.dtypes.astype(str).values,
        'n_missing': n_missing.values,
        'n_zeros': n_zeros.values,
        'complete_rate': 1 - (n_missing.values / len(df))
    })

    # Manipulación mediante Method Chaining
    var_type_missing_df = (
        desc_df
        .assign(
            n_missing_perc = lambda x: (100 * (1 - x['complete_rate'])).round(3),
            n_zeros_perc = lambda x: (100 * (x['n_zeros'] / len(df))).round(3)
        )
        .filter(['variable_type', 'variable', 'n_missing', 'n_missing_perc', 'n_zeros', 'n_zeros_perc'])
        .sort_values(['variable_type', 'n_missing'], ascending=[True, False])
    )

    return (var_type_missing_df.style
            .background_gradient(subset=['n_missing_perc'], cmap='Reds')
            .background_gradient(subset=['n_zeros_perc'], cmap='Blues'))




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


def histogram_var_target_plot(data, var, target_var = "churn", ax = None, bins = 50):
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






def plot_categoricas(df):
    # 1. Filtrar solo columnas categóricas
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    n_vars = len(cat_cols)
    
    if n_vars == 0:
        print("No se encontraron variables categóricas.")
        return

    # 2. Configurar la estructura de la cuadrícula (3 columnas)
    n_cols = 3
    n_rows = math.ceil(n_vars / n_cols)
    
    # Crear la figura y los ejes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    
    # Aplanamos el array de ejes para iterar fácilmente, 
    # manejando el caso de que solo haya una fila
    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes] # Convertir en lista para consistencia

    sns.set_style("whitegrid")

    # 3. Iterar sobre las columnas y crear los gráficos
    for i, col in enumerate(cat_cols):
        # Ordenar por frecuencia
        order = df[col].value_counts().index
        
        # Dibujar el gráfico en el subeje correspondiente
        ax = sns.countplot(data=df, x=col, order=order, ax=axes[i], color='skyblue')
        
        # Estética de cada subgráfico
        axes[i].set_title(f'Distribución de {col}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('') # Limpiamos etiqueta X para no saturar
        axes[i].set_ylabel('Frecuencia')
        
        # Rotar etiquetas si hay muchas categorías
        if df[col].nunique() > 4:
            axes[i].tick_params(axis='x', rotation=45)
            
        # Añadir el número exacto encima de cada barra
        for p in ax.patches:
            height = p.get_height()
            axes[i].annotate(f'{int(height)}', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom', fontsize=10, xytext=(0, 5),
                            textcoords='offset points')

    # 4. Eliminar subplots vacíos (si n_vars no es múltiplo de 3)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



def plot_cat_vs_target(df, target):
    # Filtrar categóricas y excluir el target
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target in cat_cols: cat_cols.remove(target)
    
    n_vars = len(cat_cols)
    if n_vars == 0: return print("No hay variables categóricas.")

    n_cols = 3
    n_rows = math.ceil(n_vars / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_vars > 1 else [axes]

    for i, col in enumerate(cat_cols):
        # Gráfico de barras agrupadas
        sns.countplot(data=df, x=col, hue=target, ax=axes[i], palette='viridis')
        
        axes[i].set_title(f'{col} vs {target}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Conteo')
        
        # Rotar etiquetas si hay muchas
        if df[col].nunique() > 3:
            axes[i].tick_params(axis='x', rotation=45)
        
        # Mover la leyenda para que no tape las barras
        axes[i].legend(title=target, loc='upper right')

    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()