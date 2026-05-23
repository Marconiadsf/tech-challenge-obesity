import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ======================================================
# Configuração da página
# ======================================================

st.set_page_config(
    page_title="Tech Challenge - Obesidade",
    page_icon="⚕️",
    layout="wide"
)


# ======================================================
# Dicionários de tradução
# ======================================================

traducao_obesidade = {
    "Insufficient_Weight": "Peso Insuficiente",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Nível I",
    "Overweight_Level_II": "Sobrepeso Nível II",
    "Obesity_Type_I": "Obesidade Tipo I",
    "Obesity_Type_II": "Obesidade Tipo II",
    "Obesity_Type_III": "Obesidade Tipo III"
}

traducao_genero = {
    "Male": "Masculino",
    "Female": "Feminino"
}

traducao_sim_nao = {
    "yes": "Sim",
    "no": "Não"
}

traducao_alcool = {
    "no": "Não consome",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente",
    "Always": "Sempre"
}

traducao_caec = {
    "no": "Não",
    "Sometimes": "Às vezes",
    "Frequently": "Frequentemente",
    "Always": "Sempre"
}

traducao_transporte = {
    "Public_Transportation": "Transporte Público",
    "Walking": "Caminhada",
    "Automobile": "Automóvel",
    "Motorbike": "Moto",
    "Bike": "Bicicleta"
}


# ======================================================
# Carregamento e treinamento do modelo
# ======================================================

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

    return modelo, X, list(X.columns), acuracia


@st.cache_data
def carregar_dados_dashboard():
    df = pd.read_csv("Obesity.csv")
    df["Obesity_PT"] = df["Obesity"].map(traducao_obesidade)
    df["Gender_PT"] = df["Gender"].map(traducao_genero)
    df["family_history_PT"] = df["family_history"].map(traducao_sim_nao)
    df["FAVC_PT"] = df["FAVC"].map(traducao_sim_nao)
    df["SMOKE_PT"] = df["SMOKE"].map(traducao_sim_nao)
    df["SCC_PT"] = df["SCC"].map(traducao_sim_nao)
    df["CAEC_PT"] = df["CAEC"].map(traducao_caec)
    df["CALC_PT"] = df["CALC"].map(traducao_alcool)
    df["MTRANS_PT"] = df["MTRANS"].map(traducao_transporte)
    return df


modelo, X_modelo, colunas_modelo, acuracia = treinar_modelo()
df_dash = carregar_dados_dashboard()


# ======================================================
# Função para importância das variáveis
# ======================================================

def obter_importancia_features(modelo, X):
    preprocessor = modelo.named_steps["preprocessor"]
    classifier = modelo.named_steps["classifier"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    ohe = preprocessor.named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_cols)

    feature_names = num_cols + list(cat_names)
    importancias = classifier.feature_importances_

    df_imp = pd.DataFrame({
        "Variável": feature_names,
        "Importância": importancias
    }).sort_values("Importância", ascending=False)

    return df_imp


# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Escolha uma página:",
    ["Predição", "Dashboard Analítico", "Sobre o Projeto"]
)

st.sidebar.divider()
st.sidebar.info(
    "Tech Challenge - Fase 4\n\n"
    "Modelo preditivo para apoio à análise do nível de obesidade."
)


# ======================================================
# Página 1 - Predição
# ======================================================

if pagina == "Predição":

    st.title("⚕️ Sistema Preditivo de Nível de Obesidade")

    st.write(
        """
        Esta aplicação utiliza Machine Learning para auxiliar a equipe médica
        na estimativa do nível de obesidade de um paciente com base em hábitos alimentares,
        histórico familiar, atividade física e estilo de vida.
        """
    )

    st.warning(
        """
        Este sistema é uma ferramenta de apoio à decisão e não substitui avaliação médica profissional.

        As variáveis **peso** e **altura** foram removidas do modelo para reduzir risco de *data leakage*,
        já que o nível de obesidade pode estar diretamente relacionado ao IMC.
        """
    )

    st.metric("Acurácia do modelo", f"{acuracia:.2%}")

    st.divider()

    st.header("Dados do paciente")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender_pt = st.selectbox("Gênero", ["Feminino", "Masculino"])
        age = st.number_input("Idade", min_value=1, max_value=100, value=25)
        family_history_pt = st.selectbox("Histórico familiar de sobrepeso?", ["Sim", "Não"])
        favc_pt = st.selectbox("Consome alimentos altamente calóricos com frequência?", ["Sim", "Não"])

    with col2:
        fcvc = st.slider("Frequência de consumo de vegetais", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        ncp = st.slider("Número de refeições principais por dia", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
        caec_pt = st.selectbox("Come entre as refeições?", ["Não", "Às vezes", "Frequentemente", "Sempre"])
        smoke_pt = st.selectbox("Fuma?", ["Sim", "Não"])

    with col3:
        ch2o = st.slider("Consumo diário de água", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        scc_pt = st.selectbox("Monitora calorias ingeridas?", ["Sim", "Não"])
        faf = st.slider("Frequência de atividade física", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        tue = st.slider("Tempo usando dispositivos tecnológicos", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

    col4, col5 = st.columns(2)

    with col4:
        calc_pt = st.selectbox("Frequência de consumo de álcool", ["Não consome", "Às vezes", "Frequentemente", "Sempre"])

    with col5:
        mtrans_pt = st.selectbox(
            "Meio de transporte principal",
            ["Transporte Público", "Caminhada", "Automóvel", "Moto", "Bicicleta"]
        )

    # Converter português de volta para os valores originais do dataset/modelo
    gender = {"Feminino": "Female", "Masculino": "Male"}[gender_pt]
    family_history = {"Sim": "yes", "Não": "no"}[family_history_pt]
    favc = {"Sim": "yes", "Não": "no"}[favc_pt]
    smoke = {"Sim": "yes", "Não": "no"}[smoke_pt]
    scc = {"Sim": "yes", "Não": "no"}[scc_pt]

    caec = {
        "Não": "no",
        "Às vezes": "Sometimes",
        "Frequentemente": "Frequently",
        "Sempre": "Always"
    }[caec_pt]

    calc = {
        "Não consome": "no",
        "Às vezes": "Sometimes",
        "Frequentemente": "Frequently",
        "Sempre": "Always"
    }[calc_pt]

    mtrans = {
        "Transporte Público": "Public_Transportation",
        "Caminhada": "Walking",
        "Automóvel": "Automobile",
        "Moto": "Motorbike",
        "Bicicleta": "Bike"
    }[mtrans_pt]

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
        predicao_pt = traducao_obesidade.get(predicao, predicao)

        st.subheader("Resultado da previsão")
        st.success(f"Nível previsto: {predicao_pt}")

        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(dados_paciente)[0]
            classes = modelo.classes_

            df_prob = pd.DataFrame({
                "Classe": [traducao_obesidade.get(c, c) for c in classes],
                "Probabilidade": probabilidades
            }).sort_values(by="Probabilidade", ascending=False)

            st.subheader("Probabilidades por classe")
            st.dataframe(df_prob, use_container_width=True)

            fig_prob = px.bar(
                df_prob,
                x="Classe",
                y="Probabilidade",
                color="Classe",
                title="Probabilidade estimada por classe"
            )
            fig_prob.update_layout(showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)


# ======================================================
# Página 2 - Dashboard Analítico
# ======================================================

elif pagina == "Dashboard Analítico":

    st.title("📊 Dashboard Analítico — Fatores Associados à Obesidade")

    st.write(
        """
        Este painel apresenta uma visão analítica dos principais padrões encontrados na base de dados,
        com foco em fatores comportamentais, histórico familiar e estilo de vida associados ao nível de obesidade.
        """
    )

    st.info(
        """
        Para tornar o modelo mais robusto, peso e altura foram retirados do treinamento.
        Portanto, a análise enfatiza fatores de comportamento e estilo de vida.
        """
    )

    # KPIs
    total_pacientes = len(df_dash)
    obesos = df_dash["Obesity"].isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]).mean() * 100
    sobrepeso = df_dash["Obesity"].isin(["Overweight_Level_I", "Overweight_Level_II"]).mean() * 100
    peso_normal = (df_dash["Obesity"] == "Normal_Weight").mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Pacientes", f"{total_pacientes:,}".replace(",", "."))
    k2.metric("Obesidade", f"{obesos:.1f}%")
    k3.metric("Sobrepeso", f"{sobrepeso:.1f}%")
    k4.metric("Peso normal", f"{peso_normal:.1f}%")
    k5.metric("Acurácia", f"{acuracia:.1%}")

    st.divider()

    # Distribuição e proporção das classes
    col1, col2 = st.columns(2)

    contagem = df_dash["Obesity_PT"].value_counts().reset_index()
    contagem.columns = ["Nível de obesidade", "Quantidade"]

    with col1:
        fig_bar = px.bar(
            contagem,
            x="Nível de obesidade",
            y="Quantidade",
            color="Nível de obesidade",
            title="Distribuição dos Níveis de Obesidade"
        )
        fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Quantidade de pacientes")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            contagem,
            names="Nível de obesidade",
            values="Quantidade",
            title="Proporção das Classes",
            hole=0.5
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Histórico familiar
    hist = pd.crosstab(
        df_dash["Obesity_PT"],
        df_dash["family_history_PT"],
        normalize="index"
    ) * 100

    hist = hist.reset_index().melt(
        id_vars="Obesity_PT",
        var_name="Histórico familiar",
        value_name="Percentual"
    )

    fig_hist = px.bar(
        hist,
        x="Obesity_PT",
        y="Percentual",
        color="Histórico familiar",
        title="Histórico Familiar × Nível de Obesidade",
        barmode="stack"
    )
    fig_hist.update_layout(xaxis_title="", yaxis_title="% dentro de cada nível")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(
        """
        **Insight:** O histórico familiar é uma variável importante para identificar maior predisposição
        aos níveis de sobrepeso e obesidade.
        """
    )

    st.divider()

    col3, col4 = st.columns(2)

    # Atividade física
    with col3:
        fig_faf = px.box(
            df_dash,
            x="Obesity_PT",
            y="FAF",
            color="Obesity_PT",
            title="Atividade Física × Nível de Obesidade"
        )
        fig_faf.update_layout(showlegend=False, xaxis_title="", yaxis_title="Frequência de atividade física")
        st.plotly_chart(fig_faf, use_container_width=True)

    # Água
    with col4:
        fig_ch2o = px.box(
            df_dash,
            x="Obesity_PT",
            y="CH2O",
            color="Obesity_PT",
            title="Consumo de Água × Nível de Obesidade"
        )
        fig_ch2o.update_layout(showlegend=False, xaxis_title="", yaxis_title="Consumo diário de água")
        st.plotly_chart(fig_ch2o, use_container_width=True)

    st.divider()

    col5, col6 = st.columns(2)

    # Alimentos calóricos
    favc = pd.crosstab(
        df_dash["Obesity_PT"],
        df_dash["FAVC_PT"],
        normalize="index"
    ) * 100

    favc = favc.reset_index().melt(
        id_vars="Obesity_PT",
        var_name="Consome alimentos calóricos",
        value_name="Percentual"
    )

    with col5:
        fig_favc = px.bar(
            favc,
            x="Obesity_PT",
            y="Percentual",
            color="Consome alimentos calóricos",
            title="Alimentos Calóricos × Nível de Obesidade",
            barmode="stack"
        )
        fig_favc.update_layout(xaxis_title="", yaxis_title="% dentro de cada nível")
        st.plotly_chart(fig_favc, use_container_width=True)

    # Transporte
    mtrans = pd.crosstab(
        df_dash["Obesity_PT"],
        df_dash["MTRANS_PT"],
        normalize="index"
    ) * 100

    mtrans = mtrans.reset_index().melt(
        id_vars="Obesity_PT",
        var_name="Meio de transporte",
        value_name="Percentual"
    )

    with col6:
        fig_mtrans = px.bar(
            mtrans,
            x="Obesity_PT",
            y="Percentual",
            color="Meio de transporte",
            title="Meio de Transporte × Nível de Obesidade",
            barmode="stack"
        )
        fig_mtrans.update_layout(xaxis_title="", yaxis_title="% dentro de cada nível")
        st.plotly_chart(fig_mtrans, use_container_width=True)

    st.divider()

    # Importância das variáveis
    st.subheader("Importância das Variáveis no Modelo")

    df_imp = obter_importancia_features(modelo, X_modelo).head(15)

    # Tradução parcial de nomes principais
    traducoes_features = {
        "Age": "Idade",
        "FCVC": "Consumo de vegetais",
        "NCP": "Refeições principais",
        "CH2O": "Consumo de água",
        "FAF": "Atividade física",
        "TUE": "Uso de tecnologia",
        "Gender": "Gênero",
        "family_history": "Histórico familiar",
        "FAVC": "Alimentos calóricos",
        "CAEC": "Come entre refeições",
        "SMOKE": "Fumo",
        "SCC": "Monitora calorias",
        "CALC": "Consumo de álcool",
        "MTRANS": "Transporte"
    }

    def traduzir_feature(nome):
        for chave, valor in traducoes_features.items():
            nome = nome.replace(chave, valor)
        nome = nome.replace("_yes", " = Sim")
        nome = nome.replace("_no", " = Não")
        nome = nome.replace("_Female", " = Feminino")
        nome = nome.replace("_Male", " = Masculino")
        nome = nome.replace("_Sometimes", " = Às vezes")
        nome = nome.replace("_Frequently", " = Frequentemente")
        nome = nome.replace("_Always", " = Sempre")
        nome = nome.replace("_Public_Transportation", " = Transporte Público")
        nome = nome.replace("_Walking", " = Caminhada")
        nome = nome.replace("_Automobile", " = Automóvel")
        nome = nome.replace("_Motorbike", " = Moto")
        nome = nome.replace("_Bike", " = Bicicleta")
        return nome

    df_imp["Variável"] = df_imp["Variável"].apply(traduzir_feature)

    fig_imp = px.bar(
        df_imp.sort_values("Importância"),
        x="Importância",
        y="Variável",
        orientation="h",
        color="Importância",
        title="Top 15 Variáveis Mais Importantes"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    st.subheader("Conclusão Executiva")

    st.success(
        """
        A análise indica que fatores como histórico familiar, hábitos alimentares,
        frequência de atividade física, consumo de água, uso de tecnologia e meio de transporte
        contribuem para a identificação dos diferentes níveis de obesidade.

        Como peso e altura foram removidos do modelo, a solução se torna mais útil como ferramenta
        de triagem baseada em comportamento e estilo de vida, apoiando a equipe médica na identificação
        de padrões de risco.
        """
    )


# ======================================================
# Página 3 - Sobre o Projeto
# ======================================================

else:

    st.title("📘 Sobre o Projeto")

    st.header("Objetivo")

    st.write(
        """
        O objetivo deste projeto é desenvolver uma solução de Machine Learning capaz de auxiliar
        profissionais de saúde na previsão do nível de obesidade de pacientes.
        """
    )

    st.header("Pipeline de Machine Learning")

    st.write(
        """
        A pipeline desenvolvida inclui:

        1. Leitura e entendimento da base de dados;
        2. Análise exploratória dos dados;
        3. Remoção de variáveis com risco de data leakage;
        4. Pré-processamento de variáveis numéricas e categóricas;
        5. Treinamento de modelos de classificação;
        6. Avaliação com acurácia, validação cruzada e matriz de confusão;
        7. Implantação do modelo em aplicação Streamlit.
        """
    )

    st.header("Prevenção de Data Leakage")

    st.write(
        """
        As variáveis `Weight` e `Height` foram removidas do modelo final, pois o nível de obesidade
        pode estar diretamente relacionado ao IMC, que utiliza peso e altura em seu cálculo.

        Com essa decisão, o modelo passa a utilizar fatores comportamentais e de estilo de vida,
        tornando a solução mais robusta e mais útil como ferramenta de apoio à triagem.
        """
    )

    st.header("Modelo Utilizado")

    st.write(
        f"""
        O modelo final utilizado foi um **Random Forest Classifier** ajustado para reduzir o risco
        de overfitting.

        Acurácia obtida no conjunto de teste: **{acuracia:.2%}**
        """
    )

    st.header("Limitações")

    st.write(
        """
        Esta aplicação não deve ser utilizada como diagnóstico médico definitivo.
        O resultado gerado pelo modelo deve ser interpretado como apoio à decisão,
        sendo necessária a avaliação clínica de profissionais de saúde.
        """
    )
