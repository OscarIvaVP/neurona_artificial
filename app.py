"""Aplicación Streamlit para entrenar una neurona lineal con datos propios."""

from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def prepare_xy(frame: pd.DataFrame, x_column: str, y_column: str) -> pd.DataFrame:
    """Selecciona X e Y, convierte a números y elimina filas no utilizables."""
    if x_column == y_column:
        raise ValueError("Selecciona columnas distintas para X e Y.")
    if x_column not in frame.columns or y_column not in frame.columns:
        raise ValueError("No se encontraron las columnas seleccionadas en los datos.")

    clean = frame[[x_column, y_column]].copy()
    clean.columns = ["x", "y"]
    clean["x"] = pd.to_numeric(clean["x"], errors="coerce")
    clean["y"] = pd.to_numeric(clean["y"], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        raise ValueError("Se necesitan al menos 3 filas con valores numéricos en X e Y.")
    return clean.reset_index(drop=True)


def split_data(n_rows: int, validation_fraction: float = 0.2, random_state: int = 42):
    """Devuelve índices aleatorios y reproducibles para entrenamiento y validación."""
    if n_rows < 3:
        raise ValueError("Se necesitan al menos 3 filas para separar los datos.")
    if not 0 < validation_fraction < 1:
        raise ValueError("La fracción de validación debe estar entre 0 y 1.")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_rows)
    n_validation = min(n_rows - 2, max(1, int(round(n_rows * validation_fraction))))
    return indices[n_validation:], indices[:n_validation]


def train_neuron(x_train, y_train, epochs: int = 1000, learning_rate: float = 0.05):
    """Entrena ŷ = wX + b con descenso de gradiente por lotes."""
    x_train = np.asarray(x_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    if len(x_train) != len(y_train) or len(x_train) < 2:
        raise ValueError("X e Y deben tener la misma longitud y al menos 2 filas.")
    if epochs < 1:
        raise ValueError("Las iteraciones deben ser al menos 1.")
    if learning_rate <= 0:
        raise ValueError("La tasa de aprendizaje debe ser mayor que cero.")

    x_mean, y_mean = float(np.mean(x_train)), float(np.mean(y_train))
    x_scale, y_scale = float(np.std(x_train)), float(np.std(y_train)) or 1.0
    if x_scale == 0:
        raise ValueError("La columna X no varía en las filas de entrenamiento.")

    z_x = (x_train - x_mean) / x_scale
    z_y = (y_train - y_mean) / y_scale
    weight_scaled = bias_scaled = 0.0
    loss_history = []
    n = len(z_x)
    for _ in range(int(epochs)):
        error = weight_scaled * z_x + bias_scaled - z_y
        weight_scaled -= learning_rate * (2.0 / n) * np.dot(error, z_x)
        bias_scaled -= learning_rate * (2.0 / n) * np.sum(error)
        updated_error = weight_scaled * z_x + bias_scaled - z_y
        loss_history.append(float(np.mean(updated_error**2) * y_scale**2))

    weight = weight_scaled * y_scale / x_scale
    bias = y_mean + y_scale * bias_scaled - weight * x_mean
    return {"weight": float(weight), "bias": float(bias), "loss_history": loss_history}


def predict(x, model):
    """Calcula ŷ = wX + b en la escala original de los datos."""
    return model["weight"] * np.asarray(x, dtype=float) + model["bias"]


def regression_metrics(y_true, y_pred):
    """Calcula MAE, RMSE y R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "R²": float(1 - np.sum(residual**2) / denominator) if denominator else float("nan"),
    }


st.set_page_config(
    page_title="Neurona lineal: X → Y",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Neurona artificial para predecir Y desde X")
st.markdown(
    "Carga un archivo Excel o CSV, selecciona las columnas de entrada y salida, "
    "y entrena una neurona lineal con la forma **ŷ = w · X + b**."
)


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes, file_name: str, sheet_name=None):
    """Lee los bytes del archivo cargado; la caché se invalida al cambiarlo."""
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(BytesIO(file_bytes))
        except UnicodeDecodeError:
            return pd.read_csv(BytesIO(file_bytes), encoding="latin-1")
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)
    raise ValueError("Formato no compatible. Carga un archivo .xlsx, .xlsm o .csv.")


uploaded_file = st.file_uploader(
    "Selecciona tu archivo de datos",
    type=["xlsx", "xlsm", "csv"],
    help="El archivo debe tener encabezados. Luego podrás elegir las columnas numéricas para X e Y.",
)

if uploaded_file is None:
    st.info("Carga un archivo para comenzar. El archivo debe incluir al menos dos columnas numéricas y tres filas válidas.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_suffix = Path(uploaded_file.name).suffix.lower()

try:
    selected_sheet = None
    if file_suffix in {".xlsx", ".xlsm"}:
        excel = pd.ExcelFile(BytesIO(file_bytes))
        selected_sheet = st.selectbox("Hoja del archivo Excel", excel.sheet_names)
    raw_data = read_uploaded_file(file_bytes, uploaded_file.name, selected_sheet)
except Exception as exc:
    st.error(f"No fue posible leer el archivo: {exc}")
    st.stop()

if raw_data.empty or len(raw_data.columns) < 2:
    st.error("El archivo debe tener encabezados y al menos dos columnas.")
    st.stop()

st.subheader("1. Datos cargados")
info_a, info_b = st.columns(2)
info_a.metric("Filas leídas", f"{len(raw_data):,}")
info_b.metric("Columnas", f"{len(raw_data.columns):,}")
st.dataframe(raw_data.head(10), use_container_width=True)

columns = list(raw_data.columns)
numeric_candidates = [
    col for col in columns if pd.to_numeric(raw_data[col], errors="coerce").notna().sum() >= 3
]
default_x = numeric_candidates[0] if numeric_candidates else columns[0]
default_y = numeric_candidates[1] if len(numeric_candidates) > 1 else columns[min(1, len(columns) - 1)]

st.subheader("2. Configura las columnas y el entrenamiento")
col_x, col_y = st.columns(2)
x_column = col_x.selectbox("Variable de entrada (X)", columns, index=columns.index(default_x))
y_column = col_y.selectbox("Variable objetivo (Y)", columns, index=columns.index(default_y))

with st.sidebar:
    st.header("Parámetros")
    validation_fraction = st.slider("Datos para validación", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    epochs = st.number_input("Iteraciones de entrenamiento", min_value=10, max_value=20000, value=1000, step=100)
    learning_rate = st.number_input(
        "Tasa de aprendizaje", min_value=0.001, max_value=1.0, value=0.05, step=0.01, format="%.3f"
    )
    random_state = st.number_input("Semilla de la separación", min_value=0, max_value=999999, value=42, step=1)

try:
    clean_data = prepare_xy(raw_data, x_column, y_column)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

dropped_rows = len(raw_data) - len(clean_data)
st.caption(f"Filas utilizables: {len(clean_data):,}. Filas descartadas por valores no numéricos o vacíos: {dropped_rows:,}.")

if clean_data["x"].nunique() < 2:
    st.error("La columna X debe tener al menos dos valores distintos para poder aprender una relación.")
    st.stop()

train_idx, validation_idx = split_data(
    len(clean_data), validation_fraction=float(validation_fraction), random_state=int(random_state)
)
x = clean_data["x"].to_numpy()
y = clean_data["y"].to_numpy()

try:
    model = train_neuron(x[train_idx], y[train_idx], epochs=int(epochs), learning_rate=float(learning_rate))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

pred_train = predict(x[train_idx], model)
pred_validation = predict(x[validation_idx], model)
train_metrics = regression_metrics(y[train_idx], pred_train)
validation_metrics = regression_metrics(y[validation_idx], pred_validation)

st.subheader("3. Resultados de la neurona")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Peso (w)", f"{model['weight']:.8g}")
metric_col2.metric("Bias (b)", f"{model['bias']:.8g}")
metric_col3.metric("R² validación", f"{validation_metrics['R²']:.4f}")
metric_col4.metric("RMSE validación", f"{validation_metrics['RMSE']:.6g}")
st.code(f"ŷ = {model['weight']:.8g} · X + ({model['bias']:.8g})", language="text")

plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    st.markdown("#### Datos reales y simulados")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x[train_idx], y[train_idx], s=25, alpha=0.62, color="#2878B5", label="Reales: entrenamiento")
    ax.scatter(x[validation_idx], y[validation_idx], s=32, alpha=0.8, color="#F28E2B", label="Reales: validación")
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 300)
    ax.plot(x_line, predict(x_line, model), color="#222222", linewidth=2.2, label="Neurona: ŷ = wX + b")
    ax.scatter(x[train_idx], pred_train, marker="x", s=25, color="#174A73", alpha=0.75, label="Simulados: entrenamiento")
    ax.scatter(x[validation_idx], pred_validation, marker="x", s=38, color="#A64B00", alpha=0.9, label="Simulados: validación")
    ax.set_xlabel(str(x_column))
    ax.set_ylabel(str(y_column))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with plot_col2:
    st.markdown("#### Error durante el entrenamiento")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(model["loss_history"]) + 1), model["loss_history"], color="#D1495B", linewidth=1.8)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Error cuadrático medio (ECM)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.subheader("4. Métricas de entrenamiento y validación")
metrics = pd.DataFrame(
    [
        {"Conjunto": "Entrenamiento", "Filas": len(train_idx), **train_metrics},
        {"Conjunto": "Validación", "Filas": len(validation_idx), **validation_metrics},
    ]
).set_index("Conjunto")
st.dataframe(metrics.style.format({"MAE": "{:.6g}", "RMSE": "{:.6g}", "R²": "{:.6f}"}), use_container_width=True)

with st.expander("Ver valores reales y simulados"):
    results = clean_data.copy()
    results["conjunto"] = "Entrenamiento"
    results.loc[validation_idx, "conjunto"] = "Validación"
    results["y_simulado"] = predict(x, model)
    results["error"] = results["y"] - results["y_simulado"]
    st.dataframe(results, use_container_width=True)

st.caption("La neurona es un modelo lineal de una entrada. La validación mide el ajuste en observaciones que no participaron en el entrenamiento.")
