import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="⚕️",
    layout="wide"
)

@st.cache_resource
def treinar_modelo():
    df = pd.read_csv("Obesity.csv")

    # Remoção de Weight e Height para reduzir risco de data leakage
    X = df.drop(columns=["Obesity", "Weight", "Height"])
    y = df["Obesity"]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)

    return modelo, list(X.columns), acuracia


modelo, colunas_modelo, acuracia = treinar_modelo()

st.title("Sistema Preditivo de Nível de Obesidade")

st.write(
    """
    Esta aplicação utiliza Machine Learning para auxiliar a equipe médica
    na previsão do nível de obesidade de um paciente com base em hábitos alimentares,
    histórico familiar, atividade física e estilo de vida.
    """
)

st.warning(
    """
    Este sistema é uma ferramenta de apoio à decisão e não substitui avaliação médica profissional.

    As variáveis Weight e Height foram removidas do modelo para reduzir risco de data leakage,
    já que o nível de obesidade pode estar diretamente relacionado ao IMC.
    """
)

st.metric("Acurácia do modelo", f"{acuracia:.2%}")

st.header("Dados do paciente")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gênero", ["Female", "Male"])
    age = st.number_input("Idade", min_value=1, max_value=100, value=25)
    family_history = st.selectbox("Histórico familiar de sobrepeso?", ["yes", "no"])
    favc = st.selectbox("Consome alimentos altamente calóricos com frequência?", ["yes", "no"])

with col2:
    fcvc = st.slider("Frequência de consumo de vegetais", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    ncp = st.slider("Número de refeições principais por dia", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
    caec = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Fuma?", ["yes", "no"])

with col3:
    ch2o = st.slider("Consumo diário de água", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    scc = st.selectbox("Monitora calorias ingeridas?", ["yes", "no"])
    faf = st.slider("Frequência de atividade física", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
    tue = st.slider("Tempo usando dispositivos tecnológicos", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

col4, col5 = st.columns(2)

with col4:
    calc = st.selectbox("Frequência de consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])

with col5:
    mtrans = st.selectbox(
        "Meio de transporte principal",
        ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
    )

dados_paciente = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "family_history": [family_history],
    "FAVC": [favc],
    "FCVC": [fcvc],
    "NCP": [ncp],
    "CAEC": [caec],
    "SMOKE": [smoke],
    "CH2O": [ch2o],
    "SCC": [scc],
    "FAF": [faf],
    "TUE": [tue],
    "CALC": [calc],
    "MTRANS": [mtrans]
})

dados_paciente = dados_paciente[colunas_modelo]

st.divider()

if st.button("Prever nível de obesidade"):
    predicao = modelo.predict(dados_paciente)[0]

    st.subheader("Resultado da previsão")
    st.success(f"Nível previsto: {predicao}")

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(dados_paciente)[0]
        classes = modelo.classes_

        df_prob = pd.DataFrame({
            "Classe": classes,
            "Probabilidade": probabilidades
        }).sort_values(by="Probabilidade", ascending=False)

        st.subheader("Probabilidades por classe")
        st.dataframe(df_prob, use_container_width=True)
        st.bar_chart(df_prob.set_index("Classe"))

st.divider()

st.subheader("Sobre o modelo")

st.write(
    """
    O modelo foi treinado com a base Obesity.csv utilizando uma pipeline de Machine Learning
    com pré-processamento de variáveis numéricas e categóricas.

    Para tornar a solução mais robusta, as variáveis Weight e Height foram removidas do modelo final,
    reduzindo o risco de data leakage. Dessa forma, a previsão se baseia principalmente em fatores
    comportamentais, histórico familiar e estilo de vida.
    """
)
