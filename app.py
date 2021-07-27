import streamlit as st
import tensorflow.keras.backend as Kb
import tensorflow.keras as K
import numpy as np
import plotly.graph_objs as go
import json


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
    return K.models.load_model("Modelos/"+selected_model+".h5", {"custom_loss": custom_loss})
    
def custom_loss(y_true, y_pred):
    loss = -(1/models_data[selected_model]["m"])*Kb.sum(5* y_true * Kb.log(Kb.abs(y_pred+1*10**-8))+ (1-y_true)*Kb.log(Kb.abs(1-y_pred+ 1*10**-8)))
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

#### Inicializa sistema de versiones de modelos.
def load_models():

    with open('model_data.json') as json_file:
        models_data = json.load(json_file)
        
    return list(models_data.keys()), models_data


models_names, models_data = load_models()


if page == "Introducción":
    st.markdown("<pre><h3 style='text-align: center; color: black; font-family: verdana;'>A.A.E.S         A.A.F.O         A.G.F         C.J.A.G         J.E.R.S</h3></pre>", unsafe_allow_html=True)
    st.markdown("<hr><h1 style='text-align: center; color: black; font-family: verdana;'>Red Neuronal Multicapa para predecir la probabilidad de defunción en casos de COVID-19</h1>", unsafe_allow_html=True)


        
    st.markdown("""<pre><p style='text-align: center; color: black; font-family: verdana;'>
Durante el transcurso de la pandemia de COVID-19 en México se han alcanzado altos niveles de ocupación hospitalaria,
lo cual ha obligado a médicos de primera línea a priorizar recursos, sin embargo, para llevar a acabo este proceso
no existe un método sistemático más allá de la experiencia humana. Se han desarrollado herramientas para calcular
el riesgo de mortalidad por COVID-19.Actualmente existen herramientas para calcular el riesgo de mortalidad por 
COVID-19, sin embargo, una de las limitantes centrales está en el sesgo de los datos, ya que en nuestro país se
cuenta con menos del 10% de casos mortales, motivo por el cual los algoritmos de aprendizaje de máquina
tradicionales suelen tener desviaciones importantes y no siempre logran asignar predicciones de manera certera.

El presente estudio aborda la creación de una herramienta para el cálculo del riesgo de mortalidad, con base
en ciertos factores de riesgo como la edad, sexo y comorbilidades mediante el uso de técnicas de aprendizaje
profundo, la cual logra comprender los patrones internos de los datos y ejecutar predicciones sin sesgos. La
arquitectura generada, logró una precisión promedio del 89%. Dada la factibilidad y la facilidad de uso, este
modelo tiene el potencial de ser utilizado tanto por hospitales para completar la decisión médica como por la
población en general para conocer su riesgo y emprender mejores hábitos.</p></pre>""", unsafe_allow_html = True)


    st.image("imgs/DescripcionFlujoNN.jpg")
    
    st.markdown("<hr>", unsafe_allow_html = True)


elif page == "Aplicación":
    st.markdown("<h1 style='text-align: center; color: black; font-family: verdana;'>Red Neuronal Multicapa para predecir la probabilidad de defunción en casos de COVID-19</h1>", unsafe_allow_html=True)
    last_plot = None
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
        st.markdown("<h3 style='text-align: center; color: black; font-family: verdana;'>Probabilidad de defunción</h3>", unsafe_allow_html=True)
        riesgo = st.empty()
    
    
    st.markdown("<hr><h1 style='text-align: center; color: black; font-family: verdana;'>Versión del Modelo</h1>",
                unsafe_allow_html = True)
    
    st.text("Última Actualización:")
    st.text(models_data[models_names[-1]]["act"] + ", ver: " + models_data[models_names[-1]]["ver"])
    
    st.markdown("<h2 style='text-align: center; color: black; font-family: verdana;'>Selecciona la versión del modelo a utilizar:</h2>", unsafe_allow_html=True)        
    
    selected_model = st.selectbox("Modelo:", models_names, index = len(models_names)-1)
    
    ## MODEL
    NN = get_model()
    
    st.text("Datos en uso desde 01 de marzo hasta el " + models_data[selected_model]['act']+ ", versión " + models_data[selected_model]['ver']+".")

    col1, col2, col3 = st.beta_columns([0.35, 0.60, 0.25])
    
    with col2:
    
        st.image("imgs/"+ selected_model +"_esquema.png", caption = "Esquema de entrenamiento del modelo", width = 550)
    
    actual_condiciones = []
    for cond in list_condiciones:
        actual_condiciones.append(cond in condiciones)
    pred_vector = np.array([sexo, edad, indigena]+actual_condiciones+[dias_sint]).reshape((1,14))
    riesgo.plotly_chart(get_gauge(NN.predict(pred_vector)[0][0]))
    
    