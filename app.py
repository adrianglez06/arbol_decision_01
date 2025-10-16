# app.py
# Aplicación Streamlit: Árbol de Decisión universal para cualquier dataset CSV
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

st.set_page_config(page_title="Árbol de Decisión Universal", layout="wide")

st.title("Árbol de Decisión universal")
st.caption("Sube tu CSV, elige la variable objetivo y entrena un árbol de decisión para clasificación o regresión.")

# Panel lateral
st.sidebar.header("1. Datos")
uploaded = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

use_sample = False
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
        df = None
        target_default = None
else:
    try:
        df = pd.read_csv(uploaded)
        target_default = None
    except Exception as e:
        st.error(f"CSV inválido: {e}")
        df = None

if df is not None:
    st.subheader("Vista previa de datos")
    st.dataframe(df.head(50), use_container_width=True)
    st.write(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")

    # Selección de objetivo y tipo de problema
    st.sidebar.header("2. Configurar objetivo")
    target_col = st.sidebar.selectbox("Columna objetivo (y)", options=df.columns, index=(list(df.columns).index(target_default) if target_default in df.columns else 0))
    features = [c for c in df.columns if c != target_col]

    # Detección automática del tipo
    y_sample = df[target_col].dropna()
    if y_sample.dtype.kind in ("i", "u", "f") and y_sample.nunique() > 15:
        inferred_task = "Regresión"
    else:
        inferred_task = "Clasificación"

    task = st.sidebar.radio("Tipo de problema", options=["Clasificación", "Regresión"], index=0 if inferred_task == "Clasificación" else 1, help="Puedes forzar el tipo si lo necesitas.")

    st.sidebar.header("3. División de datos")
    test_size = st.sidebar.slider("Tamaño de test (%)", min_value=10, max_value=50, value=20, step=5) / 100.0
    random_state = st.sidebar.number_input("random_state", min_value=0, max_value=10_000, value=42, step=1)

    st.sidebar.header("4. Hiperparámetros del árbol")
    max_depth = st.sidebar.slider("max_depth (profundidad máxima)", 1, 20, 5)
    min_samples_split = st.sidebar.slider("min_samples_split", 2, 50, 2)
    min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 50, 1)
    criterion_cls = st.sidebar.selectbox("criterion (clasificación)", ["gini", "entropy", "log_loss"], index=0)
    criterion_reg = st.sidebar.selectbox("criterion (regresión)", ["squared_error", "absolute_error", "friedman_mse", "poisson"], index=0)
    class_weight_balanced = st.sidebar.checkbox("class_weight='balanced' (solo clasificación)", value=False)

    # Preprocesado
    X = df[features].copy()
    y = df[target_col].copy()

    # Separar columnas numéricas y categóricas
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols)
        ],
        remainder="drop"
    )

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

    # Entrenamiento
    stratify_arg = y if (task == "Clasificación" and y.nunique() > 1 and y.shape[0] > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    train_btn = st.sidebar.button("Entrenar modelo")

    if train_btn:
        model.fit(X_train, y_train)

        st.success("Modelo entrenado.")
        y_pred = model.predict(X_test)

        # Métricas y reportes
        st.subheader("Evaluación")

        if task == "Clasificación":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            st.write(f"Accuracy: {acc:.4f}")
            st.write(f"F1 macro: {f1:.4f}")

            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_train))
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            im = ax_cm.imshow(cm, interpolation="nearest")
            ax_cm.set_xticks(range(len(np.unique(y_train))))
            ax_cm.set_yticks(range(len(np.unique(y_train))))
            ax_cm.set_xticklabels(np.unique(y_train), rotation=45, ha="right")
            ax_cm.set_yticklabels(np.unique(y_train))
            ax_cm.set_xlabel("Predicho")
            ax_cm.set_ylabel("Real")
            for (i, j), val in np.ndenumerate(cm):
                ax_cm.text(j, i, int(val), ha="center", va="center")
            st.pyplot(fig_cm)

            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T)
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"R²: {r2:.4f}")

            # Gráfico Predicho vs Real
            fig_sc, ax_sc = plt.subplots(figsize=(5, 4))
            ax_sc.scatter(y_test, y_pred)
            min_val = min(np.min(y_test), np.min(y_pred))
            max_val = max(np.max(y_test), np.max(y_pred))
            ax_sc.plot([min_val, max_val], [min_val, max_val])
            ax_sc.set_xlabel("Real")
            ax_sc.set_ylabel("Predicho")
            st.pyplot(fig_sc)

            # Residuos
            resid = y_pred - y_test
            fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
            ax_hist.hist(resid, bins=20)
            ax_hist.set_title("Residuos")
            st.pyplot(fig_hist)

        # Importancias de características
        st.subheader("Importancia de características")
        # Obtener nombres de columnas transformadas
        preprocessor = model.named_steps["pre"]
        feature_names_num = num_cols
        feature_names_cat = []
        if len(cat_cols) > 0:
            ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names_cat = ohe_names
        feature_names = feature_names_num + feature_names_cat

        try:
            importances = model.named_steps["model"].feature_importances_
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True)

            fig_imp, ax_imp = plt.subplots(figsize=(6, 6))
            ax_imp.barh(imp_df["feature"][:20][::-1], imp_df["importance"][:20][::-1])
            ax_imp.set_title("Top 20 importancias")
            st.pyplot(fig_imp)
        except Exception as e:
            st.info(f"No hay importancias disponibles: {e}")

        # Visualización del árbol
        st.subheader("Visualización del árbol")
        try:
            # Extraer el estimator entrenado y graficar
            tree_model = model.named_steps["model"]
            fig_tree, ax_tree = plt.subplots(figsize=(14, 10))
            plot_tree(
                tree_model,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                max_depth=max_depth
            )
            st.pyplot(fig_tree)
        except Exception as e:
            st.warning(f"No se pudo dibujar el árbol: {e}")

        # Predicciones y descarga
        st.subheader("Descargas")

        # Archivo con predicciones sobre test
        pred_df = X_test.copy()
        pred_df[target_col + "_real"] = y_test.values
        pred_df[target_col + "_predicho"] = y_pred
        csv_buf = io.StringIO()
        pred_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Descargar predicciones (CSV)",
            data=csv_buf.getvalue(),
            file_name="predicciones_test.csv",
            mime="text/csv"
        )

        # Modelo serializado
        bin_buf = io.BytesIO()
        pickle.dump(model, bin_buf)
        st.download_button(
            label="Descargar modelo (pickle)",
            data=bin_buf.getvalue(),
            file_name="modelo_arbol.pkl",
            mime="application/octet-stream"
        )

else:
    st.info("Sube un CSV o selecciona un dataset de ejemplo para empezar.")
