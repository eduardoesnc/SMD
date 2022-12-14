import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



st.set_page_config(
    page_title = "Análise de dados de seguros automotivos",
    page_icon = './assets/car-logo.png',
    layout = "wide",
    menu_items = {
        'About': "Desenvolvedores: "
                 "Anna Carolina Lopes Dias Coêlho Bejan, "
                 "Atlas Cipriano da Silva, "
                 "Eduardo Estevão Nunes Cavalcante, "
                 "Matheus Mota Fernandes"
    },
    initial_sidebar_state='expanded'
)

st.sidebar.image('./assets/logo InsuranceTech.png',caption='InsuranceTech', use_column_width=True)
st.sidebar.header('Dashboard')

# Leitura e tratamento dos dados
# @st.cache
def readData():
    dataset = pd.read_csv('./data/train.csv')
    return dataset
bf = readData()


def tratarDados(df):
    #Idade do segurado
    df['age_of_policyholder'] = round(df['age_of_policyholder'].mul(100))
    #max_torque e max_power
    df["max_torque_Nm"] = df['max_torque'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*Nm)").astype('float64')
    df["max_torque_rpm"] = df['max_torque'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')

    df["max_power_bhp"] = df['max_power'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*bhp)").astype('float64')
    df["max_power_rpm"] = df['max_power'].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')

tratarDados(bf)

#Criação de array com o nome de todas as colunas para facilitar na criação dos filtros
dict_nome_colunas = ['Idade do carro em anos', 'Idade do segurado em anos', 'Área do segurado', 'Densidade populacional',
     'Código da fabricante do carro', 'Segmento do carro (A / B1 / B2 / C1 / C2)', 'Modelo do carro',
     'Tipo de combustível usado no carro', 'Torque máximo gerado pelo carro (Nm@rpm)', 'Força máxima gerada pelo carro (bhp@rpm)',
     'Tipo de motor usado pelo carro', 'Número de airbags instalados no carro', 'Tem controle de estabilização eletrônica?',
     'O volante é ajustável?', 'Tem sistema de monitoramento da pressão do pneu?', 'Tem sensores de ré?',
     'Tem câmera de ré?', 'Tipo de freio usado no carro', 'Cilindradas do motor (cc)', 'Quantidade de cilindros do carro',
     'Tipo de transmissão do carro', 'Quantidade de marchas do carro', 'Tipo de direção do carro', 'Espaço necessário pro carro fazer uma certa curva',
     'Comprimento do carro em milímetros', 'Largura do carro em milímetros', 'Altura do carro em milímetros', 'Peso máximo suportado pelo carro',
     'Tem farol de neblina?', 'Tem limpador de vidro traseiro?', 'Tem desembaçador de vidro traseiro?', 'Tem assistência de freio?',
     'Tem trava elétrica de porta?', 'Tem direção hidráulica?', 'O acento do motorista é ajustável?', 'Tem espelho de retrovisor traseiro?',
     'Tem luz indicativa de problemas no motor?', 'Tem sistema de alerta de velocidade?', 'Classificação de segurança pela NCAP (de 0 a 5)']
nome_colunas = ['policy_id','policy_tenure','age_of_car','age_of_policyholder','area_cluster','population_density','make','segment'
    ,'model','fuel_type','max_torque','max_power','engine_type','airbags','is_esc','is_adjustable_steering','is_tpms',
    'is_parking_sensors','is_parking_camera','rear_brakes_type','displacement','cylinder','transmission_type','gear_box','steering_type',
    'turning_radius','length','width','height','gross_weight','is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer'
    ,'is_rear_window_defogger','is_brake_assist','is_power_door_locks','is_central_locking,is_power_steering',
    'is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw','is_speed_alert','ncap_rating','is_claim']

#Verificando se há valores nulos
# @st.cache
# def manterDados(bf):
#     null = bf.isnull().sum()
#     return null

st.title('Ánalise de dados com o dataset Car-Insurance')

#Download do dataset
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(bf)

st.download_button(
    label="Baixar dataset",
    data=csv,
    file_name='car-insurance.csv',
    mime='text/csv',
)
#___________________________________________________#
st.write('Fonte: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification')
st.write('Dicionário de dados: https://github.com/eduardoesnc/SMD/blob/streamlit/data/Dicionário%20de%20dados%20-%20Car%20Insurance%20Database.pdf')
st.caption('O dataset não tem valores nulos')

st.markdown("---")

if st.sidebar.checkbox('Mostrar dataset'):
    st.subheader('Dataset')
    st.dataframe(bf)

if st.sidebar.checkbox('Mostras os tipos dos dados do dataset'):
    st.subheader('Tipos dos dados')
    bf.dtypes

matplotCheck = st.sidebar.checkbox('Mostrar gráficos feitos com matplotlib')

st.sidebar.markdown("---")
# if st.checkbox('Mostrar média dos valores das colunas'):
#     st.subheader('Média dos valores das colunas')
#     st.table(manterDados(bf))

# #________________GRÁFICOS COM MATPLOTLIB_________________#
plt.rcParams["figure.figsize"] = (15,8)
if matplotCheck:
    st.header('Gráficos feitos com matplotlib')

    segment_a = bf.fuel_type.loc[bf.segment == "A"].value_counts()
    segment_b1 = bf.fuel_type.loc[bf.segment == "B1"].value_counts()
    segment_b2 = bf.fuel_type.loc[bf.segment == "B2"].value_counts()
    segment_c1 = bf.fuel_type.loc[bf.segment == "C1"].value_counts()
    segment_c2 = bf.fuel_type.loc[bf.segment == "C2"].value_counts()

    x1 = segment_a.index
    y1 = segment_a.values

    x2 = segment_b1.index
    y2 = segment_b1.values

    x3 = segment_b2.index
    y3 = segment_b2.values

    x4 = segment_c1.index
    y4 = segment_c1.values

    x5 = segment_c2.index
    y5 = segment_c2.values

    plt.bar(x1,y1, label="Segmento A", width=0.4, align='edge')
    plt.bar(x2,y2, label="Segmento B1", width=-0.4, align='edge')
    plt.bar(x3,y3, label="Segmento B2", width=0.4, align='edge')
    plt.bar(x4,y4, label="Segmento C1", width=-0.4, align='edge')
    plt.bar(x5,y5, label="Segmento C2", width=0.4, align='edge')
    plt.legend()
    st.subheader('Tipos de combustível por Segmento de carro')

    st.pyplot(plt)

    plt.clf()

    #___________________________________________#


    claim = bf.age_of_policyholder.loc[bf.is_claim == 1].value_counts()

    x1 = claim.index
    y1 = claim.values

    plt.bar(x1,y1, width=0.4, align='edge')

    plt.legend()
    st.subheader('Idades com grandes chances de reivindicar o seguro')

    st.pyplot(plt)

    plt.clf()

    notClaim = bf.age_of_policyholder.loc[bf.is_claim == 0].value_counts()

    x1 = notClaim.index
    y1 = notClaim.values

    plt.bar(x1,y1, width=-0.4, align='edge')

    plt.legend()
    st.subheader('Idades com poucas chances de reivindicar o seguro')

    st.pyplot(plt)

    plt.clf()

    # ___________________________________________#
    st.subheader('Média das idades dos carros em relação aos modelos de carros')
    bf.groupby("model")["age_of_car"].mean().sort_values().plot(kind="bar")
    plt.xlabel("Modelos")
    plt.ylabel("Média das Idades dos carros")

    st.pyplot(plt)
    plt.clf()



    #___________________________________________#

    claimChance = bf.model.loc[bf.is_claim == 1].value_counts(normalize=True)

    x = claimChance.values
    st.subheader('Qual é o modelo mais provável de reivindicar o seguro?')
    plt.pie(x, labels=["M6", "M1","M4","M8","M7","M9","M3","M5","M2","M10","M11"], autopct='%1.1f%%')

    st.pyplot(plt)
    plt.clf()

    st.markdown("---")

# ___________________________________________#

#_____________GRÁFICOS USANDO APENAS STREAMLIT________________#
st.header('Gráficos feitos com Streamlit')

option = st.selectbox(
    'Seleciona um para comparar com a possibilidade de reivindicação dentro de 6 meses:',
    dict_nome_colunas)


if option == 'Idade do carro em anos':
    st.subheader("Idade do carro em anos X is_claim")
    st.bar_chart(data=bf, x='age_of_car', y='is_claim')

elif option == 'Idade do segurado em anos':
    st.subheader("Idade do segurado em anos X is_claim")
    st.bar_chart(data=bf, x='age_of_policyholder', y='is_claim')

elif option == 'Área do segurado':
    st.subheader("Área do segurado X is_claim")
    st.bar_chart(data=bf, x='area_cluster', y='is_claim')

elif option == 'Densidade populacional':
    st.subheader("Densidade populacional X is_claim")
    st.bar_chart(data=bf, x='population_density', y='is_claim')

elif option == 'Código da fabricante do carro':
    st.subheader("Código da fabricante do carro X is_claim")
    st.bar_chart(data=bf, x='make', y='is_claim')

elif option == 'Segmento do carro (A / B1 / B2 / C1 / C2)':
    st.subheader("Segmento do carro (A / B1 / B2 / C1 / C2) X is_claim")
    st.bar_chart(data=bf, x='segment', y='is_claim')

elif option == 'Modelo do carro':
    st.subheader("Modelo do carro X is_claim")
    st.bar_chart(data=bf, x='model', y='is_claim')

elif option == 'Tipo de combustível usado no carro':
    st.subheader("Tipo de combustível usado no carro X is_claim")
    st.bar_chart(data=bf, x='fuel_type', y='is_claim')

elif option == 'Torque máximo gerado pelo carro (Nm@rpm)':
    st.subheader("Torque máximo gerado pelo carro (Nm@rpm) X is_claim")
    st.bar_chart(data=bf, x='max_torque', y='is_claim')

elif option == 'Força máxima gerada pelo carro (bhp@rpm)':
    st.subheader("Força máxima gerada pelo carro (bhp@rpm) X is_claim")
    st.bar_chart(data=bf, x='max_power', y='is_claim')

elif option == 'Tipo de motor usado pelo carro':
    st.subheader("Tipo de motor usado pelo carro X is_claim")
    st.bar_chart(data=bf, x='engine_type', y='is_claim')

elif option == 'Número de airbags instalados no carro':
    st.subheader("Número de airbags instalados no carro X is_claim")
    st.bar_chart(data=bf, x='airbags', y='is_claim')

elif option == 'Tem controle de estabilização eletrônica?':
    st.subheader("Tem controle de estabilização eletrônica? X is_claim")
    st.bar_chart(data=bf, x='is_esc', y='is_claim')

elif option == 'O volante é ajustável?':
    st.subheader("O volante é ajustável? X is_claim")
    st.bar_chart(data=bf, x='is_adjustable_steering', y='is_claim')

elif option == 'Tem sistema de monitoramento da pressão do pneu?':
    st.subheader("Tem sistema de monitoramento da pressão do pneu? X is_claim")
    st.bar_chart(data=bf, x='is_tpms', y='is_claim')

elif option == 'Tem sensores de ré?':
    st.subheader("Tem sensores de ré? X is_claim")
    st.bar_chart(data=bf, x='is_parking_sensors', y='is_claim')

elif option == 'Tem câmera de ré?':
    st.subheader("Tem câmera de ré? X is_claim")
    st.bar_chart(data=bf, x='is_parking_camera', y='is_claim')

elif option == 'Tipo de freio usado no carro':
    st.subheader("Tipo de freio usado no carro X is_claim")
    st.bar_chart(data=bf, x='rear_brakes_type', y='is_claim')

elif option == 'Cilindradas do motor (cc)':
    st.subheader("Cilindradas do motor (cc) X is_claim")
    st.bar_chart(data=bf, x='displacement', y='is_claim')

elif option == 'Quantidade de cilindros do carro':
    st.subheader("Quantidade de cilindros do carro X is_claim")
    st.bar_chart(data=bf, x='cylinder', y='is_claim')

elif option == 'Tipo de transmissão do carro':
    st.subheader("Tipo de transmissão do carro X is_claim")
    st.bar_chart(data=bf, x='transmission_type', y='is_claim')

elif option == 'Quantidade de marchas do carro':
    st.subheader("Quantidade de marchas do carro X is_claim")
    st.bar_chart(data=bf, x='gear_box', y='is_claim')

elif option == 'Tipo de direção do carro':
    st.subheader("Tipo de direção do carro X is_claim")
    st.bar_chart(data=bf, x='steering_type', y='is_claim')

elif option == 'Espaço necessário pro carro fazer uma certa curva':
    st.subheader("Espaço necessário pro carro fazer uma certa curva X is_claim")
    st.bar_chart(data=bf, x='turning_radius', y='is_claim')

elif option == 'Comprimento do carro em milímetros':
    st.subheader("Comprimento do carro em milímetros X is_claim")
    st.bar_chart(data=bf, x='length', y='is_claim')

elif option == 'Largura do carro em milímetros':
    st.subheader("Largura do carro em milímetros X is_claim")
    st.bar_chart(data=bf, x='width', y='is_claim')

elif option == 'Altura do carro em milímetros':
    st.subheader("Altura do carro em milímetros X is_claim")
    st.bar_chart(data=bf, x='height', y='is_claim')

elif option == 'Peso máximo suportado pelo carro':
    st.subheader("Peso máximo suportado pelo carro X is_claim")
    st.bar_chart(data=bf, x='gross_weight', y='is_claim')

elif option == 'Tem farol de neblina?':
    st.subheader("Tem farol de neblina? X is_claim")
    st.bar_chart(data=bf, x='is_front_fog_lights', y='is_claim')

elif option == 'Tem limpador de vidro traseiro?':
    st.subheader("Tem limpador de vidro traseiro? X is_claim")
    st.bar_chart(data=bf, x='is_rear_window_wiper', y='is_claim')

elif option == 'Tem desembaçador de vidro traseiro?':
    st.subheader("Tem desembaçador de vidro traseiro? X is_claim")
    st.bar_chart(data=bf, x='is_rear_window_defogger', y='is_claim')

elif option == 'Tem assistência de freio?':
    st.subheader("Tem assistência de freio? X is_claim")
    st.bar_chart(data=bf, x='is_brake_assist', y='is_claim')

elif option == 'Tem trava elétrica de porta?':
    st.subheader("Tem trava elétrica de porta? X is_claim")
    st.bar_chart(data=bf, x='is_power_door_lock', y='is_claim')

elif option == 'Tem trava de portas central?':
    st.subheader("Tem trava de portas central? X is_claim")
    st.bar_chart(data=bf, x='is_central_locking', y='is_claim')

elif option == 'Tem direção hidráulica?':
    st.subheader("Tem direção hidráulica? X is_claim")
    st.bar_chart(data=bf, x='is_power_steering', y='is_claim')

elif option == 'O acento do motorista é ajustável?':
    st.subheader("O acento do motorista é ajustável? X is_claim")
    st.bar_chart(data=bf, x='is_driver_seat_height_adjustable', y='is_claim')

elif option == 'Tem espelho de retrovisor traseiro?':
    st.subheader("Tem espelho de retrovisor traseiro? X is_claim")
    st.bar_chart(data=bf, x='is_day_night_rear_view_mirror', y='is_claim')

elif option == 'Tem luz indicativa de problemas no motor?':
    st.subheader("Tem luz indicativa de problemas no motor? X is_claim")
    st.bar_chart(data=bf, x='is_ecw', y='is_claim')

elif option == 'Tem sistema de alerta de velocidade?':
    st.subheader("Tem sistema de alerta de velocidade? X is_claim")
    st.bar_chart(data=bf, x='is_speed_alert', y='is_claim')

elif option == 'Classificação de segurança pela NCAP (de 0 a 5)':
    st.subheader("Classificação de segurança pela NCAP (de 0 a 5) X is_claim")
    st.bar_chart(data=bf, x='ncap_rating', y='is_claim')

st.caption('Gráficos para analisar como os valores das colunas interagem com a chance de reivindicação')
st.markdown("---")

selecoes = st.sidebar.multiselect('Escolha duas opções e o tipo de gráfico desejado', nome_colunas)
tipoGrafico = st.sidebar.radio('',['Linha', 'Barra'])
if len(selecoes) == 2:
    st.subheader("Gráficos formados pelas opções da multiseleção")
    if tipoGrafico == 'Linha':
        st.line_chart(data=bf, x=selecoes[0], y=selecoes[1])
    elif tipoGrafico == 'Barra':
        st.bar_chart(data=bf, x=selecoes[0], y=selecoes[1])
elif len(selecoes) < 2:
    st.subheader('Escolha opções na sidebar para formar um gráfico')
else:
    st.subheader('Escolha opções para formar um gráfico')
    st.error("Selecione apenas duas opções")

st.markdown("---")


numericos = bf.select_dtypes(include = [np.float64, np.int64])
