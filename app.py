import streamlit as st
import tensorflow.keras as K
import numpy as np
import plotly.graph_objs as go

st.set_page_config(
     page_title="NN COVID-19",
     layout="wide",
     page_icon = "💉"
 )

st.markdown("<h1 style='text-align: center; color: black; font-family: verdana;'>Red Neuronal Multicapa para predecir la probabilidad de defunción en casos de COVID-19</h1>", unsafe_allow_html=True)
last_plot = None

@st.cache(suppress_st_warning = True)
def get_model():
    return K.models.load_model("ModeloB.h5", {"custom_loss": custom_loss})
    

def custom_loss(y_true, y_pred):
    loss = -(1/1912715)*K.sum(5* y_true * K.log(K.abs(y_pred+1*10**-8))+ (1-y_true)*K.log(K.abs(1-y_pred+ 1*10**-8)))
    return loss


def get_gauge(pred):
    pred = round (pred*100)
    if pred>=50:
        color = "red"
    else:
        color = "orange"  
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = pred,
        gauge = {"axis": {"range": [0, 100], "dtick": 25, "showticklabels": True, "tickcolor": "red"}, 
                 "bar":{"color": color, "thickness": 0.85}},
        number = {"suffix": "%"} ))
    fig.update_layout(
        height =550,
        width = 750)
    return fig

NN = get_model()
list_condiciones = ["Diabetes", "EPOC", "Inmunosupresión","Neumonía",
                    "Embarazo", "Cardiovascular", "Renal Crónica",
                    "Tabaquismo", "Obesidad", "Hipertensión"]

col1, col2, col3 = st.beta_columns([0.25,0.15, 0.60])

with col1:
    st.markdown("<h2 style='text-align: center; color: black; font-family: verdana;'>Características del individuo</h2>", unsafe_allow_html=True)        
    sexo = st.radio("Sexo", ("Hombre", "Mujer")) == "Hombre"
    edad = st.number_input("Edad", min_value = 0, max_value = 105, step =  1)/105
    indigena = int(st.checkbox("Origen Indígena"))
    condiciones = st.multiselect("Condiciones", list_condiciones)
    dias_sint = st.number_input("Días con síntomas", min_value=0 , max_value= 15, step = 1)/15    
      
with col3:
    st.markdown("<h2 style='text-align: center; color: black; font-family: verdana;'>Probabilidad Resultante</h2>", unsafe_allow_html=True)
    riesgo = st.empty()

actual_condiciones = []
for cond in list_condiciones:
    actual_condiciones.append(cond in condiciones)
pred_vector = np.array([sexo, edad, indigena]+actual_condiciones+[dias_sint]).reshape((1,14))
riesgo.plotly_chart(get_gauge(NN.predict(pred_vector)[0][0]))

    

    
    