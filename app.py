import streamlit as st
import tensorflow.keras as K
import numpy as np
import plotly.graph_objs as go

st.set_page_config(
     page_title="NN COVID-19",
     layout="wide",
     page_icon = "💉",
     initial_sidebar_state="expanded"
 )

st.sidebar.header("Selecciona la página")
page = st.sidebar.selectbox("Página:", ["Introducción", "Aplicación"])



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

if page == "Introducción":
    st.markdown("<pre><h3 style='text-align: center; color: black; font-family: verdana;'>A.A.E.S         A.A.F.O         A.G.F         C.J.A.G         J.E.R.S</h3></pre>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black; font-family: verdana;'>Red Neuronal Multicapa para predecir la probabilidad de defunción en casos de COVID-19</h1>", unsafe_allow_html=True)


        
    st.markdown("""<pre><p style='text-align: center; color: black; font-family: verdana;'>
Durante el transcurso de la pandemia de COVID-19, se han alcanzado niveles altos de ocupación 
hospitalaria en México, obligando a los médicos de primera línea priorizar recursos hospitalarios, 
sin embargo, no existe método replicable más allá de la experiencia humana. Actualmente existen 
herramientas para calcular el riesgo de mortalidad por COVID-19, pero requieren de datos tomados 
por instrumentos médicos y no siempre logran asignar predicciones. Con el fin de brindar una 
herramienta objetiva para el análisis del riesgo de mortalidad y con ello complementar las 
decisiones médicas, presentamos la siguiente investigación cuyo propósito es proporcionar 
en procentaje el riesgo de defunción para personas contagiadas en México de COVID-19 dados 
ciertos factores de riesgo como sus comorbilidades, su edad y sexo mediante el uso de técnicas 
de Deep Learning. Nuestro modelo logró un F1 score de 0.44, siendo especialmente eficaz para 
los pacientes mayores a 40 años. Dada la factibilidad y la facilidad de uso, nuestro modelo 
podría ser utilizado para que los hospitales sean más eficaces al asignar sus recursos y 
clasificar con más certeza el estado de los pacientes.</p></pre>""", unsafe_allow_html = True)


    st.image("DescripcionFlujoNN.jpg")


elif page == "Aplicación":
    st.markdown("<h1 style='text-align: center; color: black; font-family: verdana;'>Red Neuronal Multicapa para predecir la probabilidad de defunción en casos de COVID-19</h1>", unsafe_allow_html=True)
    last_plot = None
    NN = get_model()
    list_condiciones = ["Diabetes", "EPOC", "Inmunosupresión","Neumonía",
                        "Embarazo", "Cardiovascular", "Renal Crónica",
                        "Tabaquismo", "Obesidad", "Hipertensión"]
    
    
    col1, col2, col3 = st.beta_columns([0.25,0.15, 0.60])
    
    with col1:
        st.markdown("<h2 style='text-align: center; color: black; font-family: verdana;'>Características del individuo</h2>", unsafe_allow_html=True)        
        sexo = st.radio("Sexo", ("Hombre", "Mujer"), index = 0) == "Hombre"
        edad = st.number_input("Edad", min_value = 1,value = 50, max_value = 105, step =  1)/105
        indigena = int(st.checkbox("Origen Indígena"))
        condiciones = st.multiselect("Condiciones", list_condiciones, default= ["Neumonía"])
        dias_sint = st.number_input("Días con síntomas", min_value=0 , max_value= 15, value = 5,step = 1)/15    
          
    with col3:
        st.markdown("<h2 style='text-align: center; color: black; font-family: verdana;'>Probabilidad de defunción</h2>", unsafe_allow_html=True)
        riesgo = st.empty()
    
    actual_condiciones = []
    for cond in list_condiciones:
        actual_condiciones.append(cond in condiciones)
    pred_vector = np.array([sexo, edad, indigena]+actual_condiciones+[dias_sint]).reshape((1,14))
    riesgo.plotly_chart(get_gauge(NN.predict(pred_vector)[0][0]))

    

    
    