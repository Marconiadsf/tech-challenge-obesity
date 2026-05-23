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
# CSS customizado
# ======================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f2b3d;
        margin-bottom: 8px;
    }

    .subtitle {
        font-size: 18px;
        color: #4b5563;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        color: #1f2b3d;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .info-box {
        background-color: #fff8db;
        color: #7a5c00;
        padding: 18px 22px;
        border-radius: 12px;
        border-left: 6px solid #f2c94c;
        margin-bottom: 24px;
        font-size: 16px;
    }

    .success-box {
        background-color: #e9f9ef;
        color: #14532d;
        padding: 18px 22px;
        border-radius: 12px;
        border-left: 6px solid #22c55e;
        margin-bottom: 24px;
        font-size: 16px;
    }

    .kpi-card {
        background: linear-gradient(135deg, #1f2b3d 0%, #111827 100%);
        padding: 24px 18px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.18);
        min-height: 120px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .kpi-title {
        color: #93a4c7;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 34px;
        font-weight: 850;
    }

    .model-card {
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 22px;
        border-radius: 14px;
        margin-bottom: 20px;
    }

    .small-caption {
        color: #6b7280;
        font-size: 14px;
    }

    .stButton>button {
        background-color: #1f2b3d;
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 700;
    }

    .stButton>button:hover {
        background-color: #111827;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
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

    # Remover duplicatas para manter consistência com o notebook
    df = df.drop_duplicates().reset_index(drop=True)

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
                n_estimators=300,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features="sqrt",
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

    # Remover duplicatas para manter consistência com o notebook
    df = df.drop_duplicates().reset_index(drop=True)

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


def card_kpi(titulo, valor):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{titulo}</div>
            <div class="kpi-value">{valor}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("⚕️ Navegação")

pagina = st.sidebar.radio(
    "Escolha uma página:",
    ["Predição", "Dashboard Analítico", "Sobre o Projeto"]
)

st.sidebar.divider()

st.sidebar.info(
    "Tech Challenge - Fase 4\n\n"
    "Sistema preditivo para apoio à análise do nível de obesidade."
)


# ======================================================
# Página 1 - Predição
# ======================================================

if pagina == "Predição":

    st.markdown('<div class="main-title">⚕️ Sistema Preditivo de Nível de Obesidade</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="subtitle">
        Ferramenta de apoio à decisão para estimar o nível de obesidade com base em hábitos alimentares,
        histórico familiar, atividade física e estilo de vida.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <b>Aviso importante:</b> este sistema é uma ferramenta de apoio à decisão e não substitui avaliação médica profissional.
        <br><br>
        As variáveis <b>peso</b> e <b>altura</b> foram removidas do modelo para reduzir risco de <i>data leakage</i>,
        já que o nível de obesidade pode estar diretamente relacionado ao IMC.
        </div>
        """,
        unsafe_allow_html=True
    )

    card1, card2, card3 = st.columns(3)

    with card1:
        card_kpi("Modelo", "Random Forest")

    with card2:
        card_kpi("Acurácia", f"{acuracia:.1%}")

    with card3:
        card_kpi("Variáveis usadas", f"{len(colunas_modelo)}")

    st.divider()

    st.markdown('<div class="section-title">Dados do Paciente</div>', unsafe_allow_html=True)

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
        calc_pt = st.selectbox(
            "Frequência de consumo de álcool",
            ["Não consome", "Às vezes", "Frequentemente", "Sempre"]
        )

    with col5:
        mtrans_pt = st.selectbox(
            "Meio de transporte principal",
            ["Transporte Público", "Caminhada", "Automóvel", "Moto", "Bicicleta"]
        )

    # Converter português para os valores originais do dataset/modelo
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

        st.markdown('<div class="section-title">Resultado da Previsão</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="success-box">
            <b>Nível previsto:</b> {predicao_pt}
            </div>
            """,
            unsafe_allow_html=True
        )

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

    st.markdown('<div class="main-title">📊 Dashboard Analítico — Padrões de Obesidade</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="subtitle">
        Insights sobre a distribuição dos níveis de obesidade e fatores associados ao comportamento,
        histórico familiar e estilo de vida dos pacientes.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        Para tornar o modelo mais robusto, peso e altura foram retirados do treinamento.
        Portanto, a análise enfatiza fatores comportamentais e de estilo de vida.
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPIs
    total_pacientes = len(df_dash)
    obesos = df_dash["Obesity"].isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]).mean() * 100
    sobrepeso = df_dash["Obesity"].isin(["Overweight_Level_I", "Overweight_Level_II"]).mean() * 100
    peso_normal = (df_dash["Obesity"] == "Normal_Weight").mean() * 100

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        card_kpi("Pacientes", f"{total_pacientes:,}".replace(",", "."))

    with k2:
        card_kpi("Obesidade", f"{obesos:.1f}%")

    with k3:
        card_kpi("Sobrepeso", f"{sobrepeso:.1f}%")

    with k4:
        card_kpi("Peso Normal", f"{peso_normal:.1f}%")

    with k5:
        card_kpi("Acurácia RF", f"{acuracia:.1%}")

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
        fig_bar.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Quantidade de pacientes",
            title_font_size=20
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            contagem,
            names="Nível de obesidade",
            values="Quantidade",
            title="Proporção das Classes",
            hole=0.55
        )
        fig_pie.update_layout(title_font_size=20)
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

    fig_hist.update_layout(
        xaxis_title="",
        yaxis_title="% dentro de cada nível",
        title_font_size=20
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(
        """
        <div class="model-card">
        <b>Insight:</b> o histórico familiar aparece como um dos fatores relevantes para identificar predisposição
        a níveis mais elevados de sobrepeso e obesidade.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        fig_faf = px.box(
            df_dash,
            x="Obesity_PT",
            y="FAF",
            color="Obesity_PT",
            title="Atividade Física × Nível de Obesidade"
        )
        fig_faf.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Frequência de atividade física",
            title_font_size=20
        )
        st.plotly_chart(fig_faf, use_container_width=True)

    with col4:
        fig_ch2o = px.box(
            df_dash,
            x="Obesity_PT",
            y="CH2O",
            color="Obesity_PT",
            title="Consumo de Água × Nível de Obesidade"
        )
        fig_ch2o.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Consumo diário de água",
            title_font_size=20
        )
        st.plotly_chart(fig_ch2o, use_container_width=True)

    st.divider()

    col5, col6 = st.columns(2)

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
        fig_favc.update_layout(
            xaxis_title="",
            yaxis_title="% dentro de cada nível",
            title_font_size=20
        )
        st.plotly_chart(fig_favc, use_container_width=True)

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
        fig_mtrans.update_layout(
            xaxis_title="",
            yaxis_title="% dentro de cada nível",
            title_font_size=20
        )
        st.plotly_chart(fig_mtrans, use_container_width=True)

    st.divider()

    st.markdown('<div class="section-title">Importância das Variáveis no Modelo</div>', unsafe_allow_html=True)

    df_imp = obter_importancia_features(modelo, X_modelo).head(15)

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

    fig_imp.update_layout(title_font_size=20)

    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    st.markdown('<div class="section-title">Conclusão Executiva</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="success-box">
        A análise indica que fatores como histórico familiar, hábitos alimentares,
        frequência de atividade física, consumo de água, uso de tecnologia e meio de transporte
        contribuem para a identificação dos diferentes níveis de obesidade.
        <br><br>
        Como peso e altura foram removidos do modelo, a solução se torna mais útil como ferramenta
        de triagem baseada em comportamento e estilo de vida, apoiando a equipe médica na identificação
        de padrões de risco.
        </div>
        """,
        unsafe_allow_html=True
    )


# ======================================================
# Página 3 - Sobre o Projeto
# ======================================================

else:

    st.markdown('<div class="main-title">📘 Sobre o Projeto</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="subtitle">
        Resumo metodológico da solução de Machine Learning desenvolvida para o Tech Challenge.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Objetivo</div>', unsafe_allow_html=True)

    st.write(
        """
        O objetivo deste projeto é desenvolver uma solução de Machine Learning capaz de auxiliar
        profissionais de saúde na previsão do nível de obesidade de pacientes.
        """
    )

    st.markdown('<div class="section-title">Pipeline de Machine Learning</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="model-card">
        A pipeline desenvolvida inclui:
        <br><br>
        <b>1.</b> Leitura e entendimento da base de dados;<br>
        <b>2.</b> Análise exploratória dos dados;<br>
        <b>3.</b> Remoção de duplicatas;<br>
        <b>4.</b> Remoção de variáveis com risco de data leakage;<br>
        <b>5.</b> Pré-processamento de variáveis numéricas e categóricas;<br>
        <b>6.</b> Treinamento e comparação de modelos de classificação;<br>
        <b>7.</b> Avaliação com acurácia, validação cruzada e matriz de confusão;<br>
        <b>8.</b> Implantação do modelo em aplicação Streamlit.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Prevenção de Data Leakage</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
        As variáveis <b>Weight</b> e <b>Height</b> foram removidas do modelo final, pois o nível de obesidade
        pode estar diretamente relacionado ao IMC, que utiliza peso e altura em seu cálculo.
        <br><br>
        Com essa decisão, o modelo passa a utilizar fatores comportamentais e de estilo de vida,
        tornando a solução mais robusta e mais útil como ferramenta de apoio à triagem.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Modelo Utilizado</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        card_kpi("Modelo", "Random Forest")

    with c2:
        card_kpi("Acurácia", f"{acuracia:.1%}")

    with c3:
        card_kpi("Objetivo", "Triagem")

    st.write(
        """
        O modelo final utilizado foi um **Random Forest Classifier ajustado** para reduzir o risco
        de overfitting. A escolha priorizou equilíbrio entre desempenho e capacidade de generalização.
        """
    )

    st.markdown('<div class="section-title">Limitações</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
        Esta aplicação não deve ser utilizada como diagnóstico médico definitivo.
        O resultado gerado pelo modelo deve ser interpretado como apoio à decisão,
        sendo necessária a avaliação clínica de profissionais de saúde.
        </div>
        """,
        unsafe_allow_html=True
    )
