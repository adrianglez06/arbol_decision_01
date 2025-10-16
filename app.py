# app.py
# Aplicación Streamlit: Árbol de Decisión para cualquier CSV
# Ejecuta: streamlit run app.py

import io
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error
)

st.set_page_config(page_title="Árbol de Decisión universal", layout="wide")

st.title("Árbol de Decisión universal")
st.caption("Sube un CSV, elige la variable objetivo y entrena un árbol de decisión para clasificación o regresión.")

# =========================
# 1) Carga de datos
# =========================
st.sidebar.header("1. Datos")
uploaded = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

use_sample = False
df = None
target_default = None

if uploaded is None:
    st.sidebar.markdown("O usa un dataset de ejemplo:")
    ejemplo = st.sidebar.selectbox("Dataset de ejemplo", ["Ninguno", "Iris (clasificación)", "Diabetes (regresión)"])
    if ejemplo == "Iris (clasificación)":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        df.rename(columns={"target": "objetivo"}, inplace=True)
        target_default = "objetivo"
        use_sample = True
    elif ejemplo == "Diabetes (regresión)":
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        df.rename(columns={"target": "objetivo"}, inplace=True)
        target_default = "objetivo"
        use_sample = True
else:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV inválido: {e}")
        df = None

if df is None:
    st.info("Sube un CSV o selecciona un dataset de ejemplo para empezar.")
    st.stop()

st.subheader("Vista previa de datos")
st.dataframe(df.head(50), use_container_width=True)
st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")

# =========================
# 2) Configuración del objetivo y tarea
# =========================
st.sidebar.header("2. Configurar objetivo")
target_col = st.sidebar.selectbox(
    "Columna objetivo (y)",
    options=list(df.columns),
    index=(list(df.columns).index(target_default) if (target_default is not None and target_default in df.columns) else 0)
)
features = [c for c in df.columns if c != target_col]

y_sample = df[target_col].dropna()
if y_sample.dtype.kind in ("i", "u", "f") and y_sample.nunique() > 15:
    inferred_task = "Regresión"
else:
    inferred_task = "Clasificación"

task = st.sidebar.radio("Tipo de problema", options=["Clasificación", "Regresión"],
                        index=0 if inferred_task == "Clasificación" else 1)

st.sidebar.header("3. División de datos")
test_size = st.sidebar.slider("Tamaño de test (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
random_state = st.sidebar.number_input("random_state", min_value=0, max_value=10000, value=42, step=1)

st.sidebar.header("4. Hiperparámetros del árbol")
max_depth = st.sidebar.slider("max_depth", 1, 30, 6)
min_samples_split = st.sidebar.slider("min_samples_split", 2, 200, 2)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 200, 1)
criterion_cls = st.sidebar.selectbox("criterion (clasificación)", ["gini", "entropy", "log_loss"], index=0)
criterion_reg = st.sidebar.selectbox("criterion (regresión)", ["squared_error", "absolute_error", "friedman_mse", "poisson"], index=0)
class_weight_balanced = st.sidebar.checkbox("class_weight='balanced' (solo clasificación)", value=False)

# =========================
# 3) Preprocesado
# =========================
X = df[features].copy()
y = df[target_col].copy()

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Corregido: compatibilidad entre scikit-learn ≥ 1.4 y ≤ 1.3
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # ≥ 1.4
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)         # ≤ 1.3

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

pre = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ],
    remainder="drop"
)

# =========================
# 4) Modelo
# =========================
if task == "Clasificación":
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion_cls,
        random_state=random_state,
        class_weight=("balanced" if class_weight_balanced else None)
    )
    model = Pipeline(steps=[("pre", pre), ("model", clf)])
else:
    reg = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion_reg,
        random_state=random_state
    )
    model = Pipeline(steps=[("pre", pre), ("model", reg)])

# =========================
# 5) Entrenamiento
# =========================
stratify_arg = y if (task == "Clasificación" and y.nunique() > 1 and y.shape[0] > 1) else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
)

train_btn = st.sidebar.button("Entrenar modelo")

if not train_btn:
    st.stop()

model.fit(X_train, y_train)
st.success("Modelo entrenado.")

y_pred = model.predict(X_test)

# =========================
# 6) Evaluación
# =========================
st.subheader("Evaluación")

if task == "Clasificación":
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"F1 macro: {f1:.4f}")

    labels = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    im = ax_cm.imshow(cm, interpolation="nearest")
    ax_cm.set_xticks(range(len(labels)))
    ax_cm.set_yticks(range(len(labels)))
    ax_cm.set_xticklabels(labels, rotation=45, ha="right")
    ax_cm.set_yticklabels(labels)
    ax_cm.set_xlabel("Predicho")
    ax_cm.set_ylabel("Real")
    for (i, j), val in np.ndenumerate(cm):
        ax_cm.text(j, i, int(val), ha="center", va="center")
    st.pyplot(fig_cm)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

else:
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"R²: {r2:.4f}")

    fig_sc, ax_sc = plt.subplots(figsize=(5, 4))
    ax_sc.scatter(y_test, y_pred)
    min_val = float(min(np.min(y_test), np.min(y_pred)))
    max_val = float(max(np.max(y_test), np.max(y_pred)))
    ax_sc.plot([min_val, max_val], [min_val, max_val])
    ax_sc.set_xlabel("Real")
    ax_sc.set_ylabel("Predicho")
    st.pyplot(fig_sc)

    resid = y_pred - y_test
    fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
    ax_hist.hist(resid, bins=20)
    ax_hist.set_title("Residuos")
    st.pyplot(fig_hist)

# =========================
# 7) Importancias y nombres de features
# =========================
st.subheader("Importancia de características")

def get_feature_names(preprocessor, num_cols, cat_cols):
    names = []
    if len(num_cols) > 0:
        names += num_cols
    if len(cat_cols) > 0:
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            # Fallback si la versión no soporta get_feature_names_out con prefijos
            ohe_names = []
            for c in cat_cols:
                ohe_names.append(f"{c}_encoded")
        names += ohe_names
    return names

preprocessor = model.named_steps["pre"]
feature_names = get_feature_names(preprocessor, num_cols, cat_cols)

try:
    importances = model.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    st.dataframe(imp_df, use_container_width=True)

    fig_imp, ax_imp = plt.subplots(figsize=(7, 7))
    top = imp_df.head(20).iloc[::-1]
    ax_imp.barh(top["feature"], top["importance"])
    ax_imp.set_title("Top 20 importancias")
    st.pyplot(fig_imp)
except Exception as e:
    st.info(f"No hay importancias disponibles: {e}")

# =========================
# 8) Visualización del árbol
# =========================
st.subheader("Visualización del árbol")

try:
    tree_model = model.named_steps["model"]
    fig_tree, ax_tree = plt.subplots(figsize=(14, 10))
    plot_tree(
        tree_model,
        feature_names=feature_names if len(feature_names) == getattr(tree_model, "n_features_", len(feature_names)) else None,
        filled=True,
        rounded=True,
        max_depth=max_depth
    )
    st.pyplot(fig_tree)
except Exception as e:
    st.warning(f"No se pudo dibujar el árbol: {e}")

# =========================
# 9) Descargas
# =========================
st.subheader("Descargas")

pred_df = X_test.copy()
pred_df[str(target_col) + "_real"] = y_test.values
pred_df[str(target_col) + "_predicho"] = y_pred
csv_buf = io.StringIO()
pred_df.to_csv(csv_buf, index=False)
st.download_button(
    label="Descargar predicciones (CSV)",
    data=csv_buf.getvalue(),
    file_name="predicciones_test.csv",
    mime="text/csv"
)

bin_buf = io.BytesIO()
pickle.dump(model, bin_buf)
st.download_button(
    label="Descargar modelo (pickle)",
    data=bin_buf.getvalue(),
    file_name="modelo_arbol.pkl",
    mime="application/octet-stream"
)
