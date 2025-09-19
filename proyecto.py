import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import odeint

#Título y texto:
st.title('Osciladores acoplados')

#Sliders
st.sidebar.header("Parámetros de los osciladores")
y_0=st.sidebar.slider('Posición inicial del primer oscilador', 0.1, 20.0, 5.0)
n= st.sidebar.slider('Número de osciladores', 2, 20, 2)
masas = st.sidebar.slider('Pesos', 1.0, 5.0, 3.0)
k = st.sidebar.slider('Constante de los muelles', 500.0, 1500.0, 500.0)


#Luego para la simulación
t= np.linspace(0, 10, 500)
frames = 500


#Matrices:
#---------------------------------------------------------------
def matriz_de_masas(n, masas):
    return masas * np.eye(n)

Ms = matriz_de_masas(n, masas)
#---------------------------------------------------------------
def matriz_de_rigidez(n, k):
    """
    Versión compacta usando numpy.diag
    """
    # Diagonal principal: 2k para todos los elementos
    diagonal_principal = 2 * k * np.ones(n)
    
    # Diagonales adyacentes: -k para n-1 elementos
    diagonales_adyacentes = -k * np.ones(n - 1)
    
    # Construir la matriz tridiagonal
    return np.diag(diagonal_principal) + \
           np.diag(diagonales_adyacentes, k=1) + \
           np.diag(diagonales_adyacentes, k=-1)
Km= matriz_de_rigidez(n, k)
#Vector posición
#---------------------------------------------------------------
def posiciones_iniciales(n, y_0):
    """
    Vector de estado completo: [y1, y2, ..., yn, vy1, vy2, ..., vyn]^T
    """
    # Posiciones: [y_0, 0, 0, ..., 0]
    posiciones = np.zeros(n)
    posiciones[0] = y_0
    
    # Velocidades: [0, 0, 0, ..., 0] (todos en reposo)
    velocidades = np.zeros(n)
    
    return np.concatenate([posiciones, velocidades])
y = posiciones_iniciales(n, y_0)

#Velocidad y aceleración, esta función ya define el sistema dinámico a simular:
#---------------------------------------------------------------
def derivadas_sistema(y, t, Ms, Km):
    num_filas = len(Ms) #devuelvo el número de filas, es decir, de osciladores
    x = y[:num_filas] #extraigo las posiciones actuales
    v = y[num_filas:2*num_filas] #concateno las velocidades al vector posicion, por eso el 2n
    aceleraciones = np.linalg.solve(Ms, -Km @ x) #linlag.solve resuelve la ecuación matricial de la segunda ley de newton del sistema a=−(M^-1)*K*x
    return np.concatenate([v, aceleraciones])
#---------------------------------------------------------------

#Aquí ya sacamos la solución del sistema que es la aceleración, odeint resuelve la ecuación diferencial, le pasamos la función de derivadas
#devuelve un vector solución que contiene al vector velocidad y el vector aceleración 
solucion = odeint(derivadas_sistema, y, t, args=(Ms, Km))

y_t = solucion[:, :n] #Vector posición dinámico y(t)
v_t = solucion[:, n:2*n] #vector velocidad dinámico y˙​(t)

#Animación--------------------------------------------------------------------------------------------------------------------------------------------------------------
idx = np.linspace(0, len(t)-1, frames).astype(int) #genera frames valores entre 0 y len(t)-1 y convierte los valores generados por linspace (que son flotantes por defecto) a enteros
posiciones_anim = y_t[idx]

x_coords = np.arange(1, n+1) #Genera números desde 1 hasta n, se le pone n+1 porque el .arange es abierto por la derecha, esto sirve como posición de cada oscilador en el eje X

#Parte de los objetos, los círculos que son los osciladores y las líneas que son los muelles
fig = go.Figure(
     data=[go.Scatter(x=x_coords, y=posiciones_anim[0], mode="lines",
                      line=dict(color="blue",width=(k/100)),
                            name="Muelles"),
           go.Scatter(x=x_coords, y=posiciones_anim[0], mode="markers",
                      marker=dict(size=(15 + 3*masas),color="red"),
                             name="Osciladores")],
    #Parte de la gráfica, el menú
    layout=go.Layout(
        xaxis=dict(range=[0, n+1]),
        yaxis=dict(range=[-y_0-2, y_0+2]),
        updatemenus=[dict(type="buttons",
                            buttons=[dict(label="Empezar",
                                        method="animate",
                                        args=[None, {"frame": {"duration": 100, "redraw": True},"fromcurrent": True}]),                                       
                                        dict(label="Pausar",
                                        method="animate",
                                           args=[[None], {"frame": {"duration": 0, "redraw": False},"mode": "immediate","transition": {"duration": 0}}])])]
    ),

    #Parte de la animación dinámica, los frames; hay que actualizar la graficación de los objetos
    frames=[go.Frame(
           data=[go.Scatter(x=x_coords, y=pos, mode="lines",
                           line=dict(color="blue", width=(k/100)),
                           name="Muelles"),
                go.Scatter(x=x_coords, y=pos, mode="markers",
                          marker=dict(size=(15+ 3*masas), color="red"),
                          name="Osciladores")])

          for pos in posiciones_anim]
    )
st.plotly_chart(fig, use_container_width=True,
                config = {
                    "displayModeBar": True,
                    "scrollZoom": False,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["zoom", "zoomIn", "zoomOut", "pan", "resetScale", "autoScale2d","toImage","lasso2d","select2d"]
})