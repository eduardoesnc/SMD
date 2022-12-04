import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title = "Teste do Streamlit pelo grupo SMD",
    layout = "wide",
    menu_items = {
        'About': "TESTES"
    }
)

@st.cache
def readData():
    dataset = pd.read_csv('./data/train.csv')
    # Normalizar valores da idade do segurado
    dataset['age_of_policyholder'] = round(dataset['age_of_policyholder'].mul(100))
    return dataset
bf = readData()

#Para não calcular toda vez que ativar o checkbox
@st.cache
def manterDados():
    avrg = bf.mean()
    return avrg

st.title('Ánalise de dados com o dataset Car-Insurance')

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

st.write('Fonte: https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification')
st.write('Dicionário de dados: https://github.com/eduardoesnc/SMD/blob/streamlit/data/Dicionário%20de%20dados%20-%20Car%20Insurance%20Database.pdf')


if st.checkbox('Mostrar dataset'):
    st.subheader('Dataset')
    st.dataframe(bf)

if st.checkbox('Mostrar média dos valores das colunas'):
    st.subheader('Média dos valores das colunas')
    st.table(manterDados())



#_____________GRÁFICOS USANDO APENAS STREAMLIT________________#

option = st.selectbox(
    'Seleciona um para comparar com a possibilidade de reivindicação dentro de 6 meses:',
    ('Idade do carro em anos', 'Idade do segurado em anos', 'Área do segurado', 'Densidade populacional',
     'Código da fabricante do carro', 'Segmento do carro (A / B1 / B2 / C1 / C2)', 'Modelo do carro',
     'Tipo de combustível usado no carro', 'Torque máximo gerado pelo carro (Nm@rpm)', 'Força máxima gerada pelo carro (bhp@rpm)',
     'Tipo de motor usado pelo carro', 'Número de airbags instalados no carro', 'Tem controle de estabilização eletrônica?',
     'O volante é ajustável?', 'Tem sistema de monitoramento da pressão do pneu?', 'Tem sensores de ré?',
     'Tem câmera de ré?', 'Tipo de freio usado no carro', 'Cilindradas do motor (cc)', 'Quantidade de cilindros do carro',
     'Tipo de transmissão do carro', 'Quantidade de marchas do carro', 'Tipo de direção do carro', 'Espaço necessário pro carro fazer uma certa curva',
     'Comprimento do carro em milímetros', 'Largura do carro em milímetros', 'Altura do carro em milímetros', 'Peso máximo suportado pelo carro',
     'Tem farol de neblina?', 'Tem limpador de vidro traseiro?', 'Tem desembaçador de vidro traseiro?', 'Tem assistência de freio?',
     'Tem trava elétrica de porta?', 'Tem direção hidráulica?', 'O acento do motorista é ajustável?', 'Tem espelho de retrovisor traseiro?',
     'Tem luz indicativa de problemas no motor?', 'Tem sistema de alerta de velocidade?', 'Classificação de segurança pela NCAP (de 0 a 5)'))


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
    st.bar_chart(data=bf, x='population density', y='is_claim')

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

#________________GRÁFICOS COM MATPLOTLIB_________________#
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
plt.title('Tipos de combustível por Segmento de carro')

st.pyplot(plt)

plt.clf()

#___________________________________________#


claim = bf.age_of_policyholder.loc[bf.is_claim == 1].value_counts()

x1 = claim.index
y1 = claim.values

plt.bar(x1,y1, width=0.4, align='edge')

plt.legend()
plt.title('Idades com grandes chances de reivindicar o seguro')

st.pyplot(plt)

plt.clf()

notClaim = bf.age_of_policyholder.loc[bf.is_claim == 0].value_counts()

x1 = notClaim.index
y1 = notClaim.values

plt.bar(x1,y1, width=-0.4, align='edge')

plt.legend()
plt.title('Idades com poucas chances de reivindicar o seguro')

st.pyplot(plt)

plt.clf()

#___________________________________________#

claimChance = bf.model.loc[bf.is_claim == 1].value_counts(normalize=True)

x = claimChance.values
plt.title('Dentre todos os carros que tem grande chance de solicitar seguro dentro dos próximos 6 meses, qual é o modelo mais provável?')
plt.pie(x, labels=["M6", "M1","M4","M8","M7","M9","M3","M5","M2","M10","M11"], autopct='%1.1f%%')

st.pyplot(plt)
