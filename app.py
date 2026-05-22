import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="⚕️",
    layout="wide"
)

@st.cache_resource
def carregar_modelo():
    modelo = joblib.load("modelo_obesidade.pkl")
    colunas_modelo = joblib.load("colunas_modelo.pkl")
    return modelo, colunas_modelo

modelo, colunas_modelo = carregar_modelo()

st.title("Sistema Preditivo de Nível de Obesidade")
st.write(
    """
    Esta aplicação utiliza Machine Learning para auxiliar a equipe médica
    na previsão do nível de obesidade de um paciente com base em dados físicos,
    hábitos alimentares, histórico familiar e estilo de vida.
    """
)

st.warning(
    "Este sistema é uma ferramenta de apoio à decisão e não substitui avaliação médica profissional."
)

st.header("Dados do paciente")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gênero", ["Female", "Male"])
    age = st.number_input("Idade", min_value=1, max_value=100, value=25)
    height = st.number_input("Altura em metros", min_value=1.00, max_value=2.50, value=1.70, step=0.01)
    weight = st.number_input("Peso em kg", min_value=20.0, max_value=250.0, value=70.0, step=0.1)

with col2:
    family_history = st.selectbox("Histórico familiar de sobrepeso?", ["yes", "no"])
    favc = st.selectbox("Consome alimentos altamente calóricos com frequência?", ["yes", "no"])
    fcvc = st.slider("Frequência de consumo de vegetais", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    ncp = st.slider("Número de refeições principais por dia", min_value=1.0, max_value=4.0, value=3.0, step=0.1)

with col3:
    caec = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Fuma?", ["yes", "no"])
    ch2o = st.slider("Consumo diário de água", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    scc = st.selectbox("Monitora calorias ingeridas?", ["yes", "no"])

col4, col5, col6 = st.columns(3)

with col4:
    faf = st.slider("Frequência de atividade física", min_value=0.0, max_value=3.0, value=1.0, step=0.1)

with col5:
    tue = st.slider("Tempo usando dispositivos tecnológicos", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

with col6:
    calc = st.selectbox("Frequência de consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox(
        "Meio de transporte principal",
        ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
    )

# DataFrame usado pelo modelo
dados_paciente = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
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

# Garantir mesma ordem das colunas usadas no treino
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
    O modelo foi treinado com a base Obesity.csv. A pipeline inclui pré-processamento
    de variáveis numéricas e categóricas, além de um classificador Random Forest
    ajustado para reduzir risco de overfitting.
    """
)
