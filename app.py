import streamlit as st
from catboost import CatBoostRegressor
import pandas as pd
import pickle
import yaml
from model_training import train_model


@st.cache
def model_loader(
    model_path="./model/model.cbm",
    config_path="./config/config.yml",
    pipeline_path="./model/data_pipeline.pkl",
):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = CatBoostRegressor()
    model.load_model(model_path)
    with open(pipeline_path, "rb") as f:
        data_pipeline = pickle.load(f)
    return config, data_pipeline, model


config, data_pipeline, model = model_loader()

df = pd.read_csv("./data/model_input.csv")
bairros = df["bairro"].unique()

st.title("Calculadora de imóveis")
st.subheader("Entre com as características do seu imóvel")

bairro = st.sidebar.selectbox(
    "Nome do bairro",
    options=bairros,
)

area = st.sidebar.number_input(
    label="Área do apto", min_value=10, max_value=500, value=70, step=25
)
quartos = st.sidebar.slider(label="# quartos", min_value=1, max_value=5, value=2)
banheiros = st.sidebar.slider(label="# banheiros", min_value=1, max_value=5, value=2)
garagens = st.sidebar.slider(label="# garagens", min_value=0, max_value=5, value=2)


input_data = pd.DataFrame(
    [
        [
            area,
            quartos,
            banheiros,
            garagens,
            bairro,
        ]
    ],
    columns=config["model_input"]["numerical_features"]
    + config["model_input"]["categorical_features"],
)
normalized_input_data = data_pipeline.transform(input_data)
preco = model.predict(normalized_input_data)[0]

if st.button(label="Calcular"):
    st.subheader(f"Preço estimado: R$ {preco:,.2f}")
    with open("memoria.csv", "a") as f:
        f.writelines(f"{bairro},{area},{quartos},{banheiros},{garagens},{preco}\n")

if st.button(label="Re-treinar"):
    training_results = train_model()
    st.image("./report/residuals.png")
    st.image("./report/shap.png")
    st.table(pd.DataFrame(training_results.items(), columns=["metric", "value"]))
    feature_importance = pd.read_csv("./report/feature_importance.csv")
    st.table(feature_importance)