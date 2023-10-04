# App de streamlit: Objetivo: EDA de variables + Cofiguración de modelos

from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from utils import create_eda_plot
import streamlit as st

# Constantes
PROJECT_ID = "mediamonks-clustering-product"
TABLA = "mediamonks-clustering-product.mmcp_raw_data_us.raw_1688520168"


# Page config
st.set_page_config(page_title="Pagina de DBS", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

# Functión para leer tabla, para poder guardarla en cache
@st.cache_data
def read_table(tabla,key_file):
    #credentials_bq = service_account.Credentials.from_service_account_file(key_file)
    client_bq = bigquery.Client(project=PROJECT_ID) #,credentials = credentials_bq)
    df_raw = client_bq.query(f"SELECT * FROM `{tabla}`").to_dataframe()
    return df_raw
   

#Test streamlit
st.title("Prueba - app")

st.write(f"tabla: {TABLA}")

# Leer tabla en bq

# Credenciales usando Service Account: credenciales
key_file = 'mediamonks-clustering-product-57781efbdc14.json'

df_raw = read_table(TABLA,key_file)

# Crea las credenciales de la cuenta de servicio
#credentials_bq = service_account.Credentials.from_service_account_file(key_file)
#client_bq = bigquery.Client(project=PROJECT_ID,credentials = credentials_bq)
#df_raw = client_bq.query(f"SELECT * FROM `{TABLA}`").to_dataframe()

########### Variables que el usuario puede elegir ###############
#@markdown Filtros

#@markdown ¿Querés trabajar sólo con usuarios con más de una compra?

remove_users_with_one_transaction = st.toggle('¿Querés trabajar sólo con usuarios con más de una compra?')
#remove_users_with_one_transaction = False #@param {type:"boolean"}

#@markdown ¿Querés quitar del análisis algún usuario que considerás outlier?

remove_users = st.toggle('¿Querés quitar del análisis algún usuario que considerás outlier?')
print(remove_users)
if remove_users:

    col1_a, col2_a, col3_a = st.columns(3)

    with col1_a:
        when_variable = st.selectbox(
        "Sobre qué variable desea aplicar el filtro?",
        ('monetary_total_money_spent', 'monetary_avg_spent_per_transaction','frequency_amount_of_transactions','frequency_avg_days_betweeen_purchases','rencency_days_since_last_purchase','rencency_days_first_last_purchase'),
        index=None,
        placeholder="Elegir variable para descartar usuarios...",
        )

    with col2_a:
        is_ = st.selectbox(
        "Qué condición?",
        ('less than', 'greather than'),
        index=None,
        placeholder="Elegir la condición...",
        )

    with col3_a:
        value = st.number_input(label="Ingrese el valor", value=1000)
        #st.header("Recency")
        #st.plotly_chart(fig3)
    
    
    #when_variable = "monetary_total_money_spent" #@param ['monetary_total_money_spent', 'monetary_avg_spent_per_transaction','frequency_amount_of_transactions','frequency_avg_days_betweeen_purchases','rencency_days_since_last_purchase','rencency_days_first_last_purchase']

    #is_ = "greater than" #@param ["less than", "greater than"]
    #value = 10000 #@param {type:"number"}


#########


# Make a copy of df to work with
df = df_raw.copy(deep = True)

# Eliminamos usuarios con sólo una compra si esa opción estaba elegida
if remove_users_with_one_transaction:
  df = df[df.frequency_amount_of_transactions>1]

# En caso de corresponder, elimino valores
if remove_users:
  if is_ == "less than":
    df = df[df[when_variable]>value]
  else:
    df = df[df[when_variable]<value]


dfm = df[['user_id','monetary_total_money_spent','monetary_avg_spent_per_transaction']]
dff = df[['user_id','frequency_amount_of_transactions','frequency_avg_days_betweeen_purchases']]
dfr = df[['user_id','rencency_days_since_last_purchase']]

fig1 = create_eda_plot(dfm,variable='m')
fig2 = create_eda_plot(dff,variable='f')
fig3 = create_eda_plot(dfr,variable='r')

import streamlit as st

col1, col2, col3 = st.columns(3)

with col1:
   st.header("Monetary")
   st.plotly_chart(fig1)

with col2:
   st.header("Frequency")
   st.plotly_chart(fig2)

with col3:
   st.header("Recency")
   st.plotly_chart(fig3)



#fig1.show()
#fig2.show()
#fig3.show()



##################

#Test streamlit
st.title("Seleccion de variables para modelo")

st.markdown('''
  Este proceso tiene dos partes: primero, cada variable (R, F, M) se divide en binnes (por defecto, 5 binnes en cada variable)
  Estos binnes son llamado G0, G1, G2, G3, G4, G5.
  Luego, los binnes de las tres variables se combinan para formar los segmentos RFM, siguiendo la lógica que está debajo en la sección "Agrupación de binnes RFM en segmentos".
            ''')
st.write(f"tabla: {TABLA}")

#@markdown ## Generar variables para el JSON -> cálculo estadístico de segmentos RFM

# Seleccion de variables

#@markdown En esta celda podés elegir qué variable R, F y M querés usar para cgenerar los segmentos de usuarios.



#@markdown

var_m = "monetary_avg_spent_per_transaction" #@param ['monetary_total_money_spent', 'monetary_avg_spent_per_transaction']
var_f = "frequency_avg_days_betweeen_purchases" #@param ['frequency_amount_of_transactions',   'frequency_avg_days_betweeen_purchases',       'rencency_days_since_last_purchase' ]
var_r = "rencency_days_since_last_purchase" #@param   ['rencency_days_since_last_purchase']


#@markdown

#@markdown ---

#@markdown *Agrupación de binnes RFM en segmentos (solo cambiar si se considera necesario)*



import plotly.graph_objects as go
#@markdown  Segment 1 - Lost
r1_start = 3 #@param [0,1,2,3,4]
r1_end = 4 #@param [0,1,2,3,4]
f1_start = 0 #@param [0,1,2,3,4]
f1_end = 1 #@param [0,1,2,3,4]
m1_start = 0 #@param [0,1,2,3,4]
m1_end = 1 #@param [0,1,2,3,4]
#@markdown  Segment 2 - Need attention
r2_start = 3 #@param [0,1,2,3,4]
r2_end = 4 #@param [0,1,2,3,4]
f2_start = 0#@param [0,1,2,3,4]
f2_end = 1#@param [0,1,2,3,4]
m2_start = 2#@param [0,1,2,3,4]
m2_end = 4#@param [0,1,2,3,4]
#@markdown  Segment 3 - Past customers
r3_start = 3#@param [0,1,2,3,4]
r3_end = 4#@param [0,1,2,3,4]
f3_start = 2#@param [0,1,2,3,4]
f3_end = 4#@param [0,1,2,3,4]
m3_start = 0#@param [0,1,2,3,4]
m3_end = 1#@param [0,1,2,3,4]
#@markdown  Segment 4 - At risk
r4_start = 3#@param [0,1,2,3,4]
r4_end = 4#@param [0,1,2,3,4]
f4_start = 2#@param [0,1,2,3,4]
f4_end = 4#@param [0,1,2,3,4]
m4_start = 2#@param [0,1,2,3,4]
m4_end = 4#@param [0,1,2,3,4]

#@markdown  Segment 5 - New customers
r5_start = 0#@param [0,1,2,3,4]
r5_end = 2#@param [0,1,2,3,4]
f5_start = 0#@param [0,1,2,3,4]
f5_end = 1#@param [0,1,2,3,4]
m5_start = 0#@param [0,1,2,3,4]
m5_end = 1#@param [0,1,2,3,4]
#@markdown  Segment 6 - Potential Loyaltist
r6_start = 0#@param [0,1,2,3,4]
r6_end = 2#@param [0,1,2,3,4]
f6_start = 0#@param [0,1,2,3,4]
f6_end = 1#@param [0,1,2,3,4]
m6_start = 2#@param [0,1,2,3,4]
m6_end = 4#@param [0,1,2,3,4]
#@markdown  Segment 7 - Low Spenders
r7_start = 0#@param [0,1,2,3,4]
r7_end = 2#@param [0,1,2,3,4]
f7_start = 2#@param [0,1,2,3,4]
f7_end = 4#@param [0,1,2,3,4]
m7_start = 0#@param [0,1,2,3,4]
m7_end = 1#@param [0,1,2,3,4]
#@markdown  Segment 8 - Loyal
r8_start = 0#@param [0,1,2,3,4]
r8_end = 2#@param [0,1,2,3,4]
f8_start = 2#@param [0,1,2,3,4]
f8_end = 4#@param [0,1,2,3,4]
m8_start = 2#@param [0,1,2,3,4]
m8_end = 4#@param [0,1,2,3,4]
#@markdown  Segment 9 - Champion
r9_start = 0#@param [0,1,2,3,4]
r9_end = 1#@param [0,1,2,3,4]
f9_start = 3#@param [0,1,2,3,4]
f9_end = 4#@param [0,1,2,3,4]
m9_start = 3#@param [0,1,2,3,4]
m9_end = 4#@param [0,1,2,3,4]

r1 = [r1_start - 0.5, r1_end + 0.5, r1_end + 0.5, r1_start - 0.5, r1_start - 0.5, r1_end + 0.5, r1_end + 0.5, r1_start - 0.5]
f1 = [f1_start - 0.5, f1_start - 0.5, f1_end + 0.5, f1_end + 0.5, f1_start - 0.5, f1_start - 0.5, f1_end + 0.5, f1_end + 0.5]
m1 = [m1_start - 0.5, m1_start - 0.5, m1_start - 0.5, m1_start - 0.5, m1_end + 0.5, m1_end + 0.5, m1_end + 0.5, m1_end + 0.5]

r2 = [r2_start - 0.5, r2_end + 0.5, r2_end + 0.5, r2_start - 0.5, r2_start - 0.5, r2_end + 0.5, r2_end + 0.5, r2_start - 0.5]
f2 = [f2_start - 0.5, f2_start - 0.5, f2_end + 0.5, f2_end + 0.5, f2_start - 0.5, f2_start - 0.5, f2_end + 0.5, f2_end + 0.5]
m2 = [m2_start - 0.5, m2_start - 0.5, m2_start - 0.5, m2_start - 0.5, m2_end + 0.5, m2_end + 0.5, m2_end + 0.5, m2_end + 0.5]

r3 = [r3_start - 0.5, r3_end + 0.5, r3_end + 0.5, r3_start - 0.5, r3_start - 0.5, r3_end + 0.5, r3_end + 0.5, r3_start - 0.5]
f3 = [f3_start - 0.5, f3_start - 0.5, f3_end + 0.5, f3_end + 0.5, f3_start - 0.5, f3_start - 0.5, f3_end + 0.5, f3_end + 0.5]
m3 = [m3_start - 0.5, m3_start - 0.5, m3_start - 0.5, m3_start - 0.5, m3_end + 0.5, m3_end + 0.5, m3_end + 0.5, m3_end + 0.5]

r4 = [r4_start - 0.5, r4_end + 0.5, r4_end + 0.5, r4_start - 0.5, r4_start - 0.5, r4_end + 0.5, r4_end + 0.5, r4_start - 0.5]
f4 = [f4_start - 0.5, f4_start - 0.5, f4_end + 0.5, f4_end + 0.5, f4_start - 0.5, f4_start - 0.5, f4_end + 0.5, f4_end + 0.5]
m4 = [m4_start - 0.5, m4_start - 0.5, m4_start - 0.5, m4_start - 0.5, m4_end + 0.5, m4_end + 0.5, m4_end + 0.5, m4_end + 0.5]

r5 = [r5_start - 0.5, r5_end + 0.5, r5_end + 0.5, r5_start - 0.5, r5_start - 0.5, r5_end + 0.5, r5_end + 0.5, r5_start - 0.5]
f5 = [f5_start - 0.5, f5_start - 0.5, f5_end + 0.5, f5_end + 0.5, f5_start - 0.5, f5_start - 0.5, f5_end + 0.5, f5_end + 0.5]
m5 = [m5_start - 0.5, m5_start - 0.5, m5_start - 0.5, m5_start - 0.5, m5_end + 0.5, m5_end + 0.5, m5_end + 0.5, m5_end + 0.5]

r6 = [r6_start - 0.5, r6_end + 0.5, r6_end + 0.5, r6_start - 0.5, r6_start - 0.5, r6_end + 0.5, r6_end + 0.5, r6_start - 0.5]
f6 = [f6_start - 0.5, f6_start - 0.5, f6_end + 0.5, f6_end + 0.5, f6_start - 0.5, f6_start - 0.5, f6_end + 0.5, f6_end + 0.5]
m6 = [m6_start - 0.5, m6_start - 0.5, m6_start - 0.5, m6_start - 0.5, m6_end + 0.5, m6_end + 0.5, m6_end + 0.5, m6_end + 0.5]

r7 = [r7_start - 0.5, r7_end + 0.5, r7_end + 0.5, r7_start - 0.5, r7_start - 0.5, r7_end + 0.5, r7_end + 0.5, r7_start - 0.5]
f7 = [f7_start - 0.5, f7_start - 0.5, f7_end + 0.5, f7_end + 0.5, f7_start - 0.5, f7_start - 0.5, f7_end + 0.5, f7_end + 0.5]
m7 = [m7_start - 0.5, m7_start - 0.5, m7_start - 0.5, m7_start - 0.5, m7_end + 0.5, m7_end + 0.5, m7_end + 0.5, m7_end + 0.5]

r8 = [r8_start - 0.5, r8_end + 0.5, r8_end + 0.5, r8_start - 0.5, r8_start - 0.5, r8_end + 0.5, r8_end + 0.5, r8_start - 0.5]
f8 = [f8_start - 0.5, f8_start - 0.5, f8_end + 0.5, f8_end + 0.5, f8_start - 0.5, f8_start - 0.5, f8_end + 0.5, f8_end + 0.5]
m8 = [m8_start - 0.5, m8_start - 0.5, m8_start - 0.5, m8_start - 0.5, m8_end + 0.5, m8_end + 0.5, m8_end + 0.5, m8_end + 0.5]

r9 = [r9_start - 0.5, r9_end + 0.5, r9_end + 0.5, r9_start - 0.5, r9_start - 0.5, r9_end + 0.5, r9_end + 0.5, r9_start - 0.5]
f9 = [f9_start - 0.5, f9_start - 0.5, f9_end + 0.5, f9_end + 0.5, f9_start - 0.5, f9_start - 0.5, f9_end + 0.5, f9_end + 0.5]
m9 = [m9_start - 0.5, m9_start - 0.5, m9_start - 0.5, m9_start - 0.5, m9_end + 0.5, m9_end + 0.5, m9_end + 0.5, m9_end + 0.5]


fig = go.Figure(data=[
    go.Mesh3d(
        x=r1,
        y=f1,
        z=m1,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#DC143C',
        flatshading=True,
        name='Lost'
    ),
    go.Mesh3d(
        x=r2,
        y=f2,
        z=m2,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#DC473C',
        flatshading=True,
        name = "Need Attention"
    ),
    go.Mesh3d(
        x=r3,
        y=f3,
        z=m3,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#00FFFF',
        flatshading=True,
        name = "Past customers"
    ),
    go.Mesh3d(
        x=r4,
        y=f4,
        z=m4,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#FFFF00',
        flatshading=True,
        name='At risk'
    ),
    go.Mesh3d(
        x=r5,
        y=f5,
        z=m5,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#00FF00',
        flatshading=True,
        name = "New Customers"
    ),
    go.Mesh3d(
        x=r6,
        y=f6,
        z=m6,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#FF00FF',
        flatshading=True,
        name = "Potential loyaltist"
    ),
    go.Mesh3d(
        x=r7,
        y=f7,
        z=m7,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#FFA500',
        flatshading=True,
        name = "Low spenders"
    ),
    go.Mesh3d(
        x=r8,
        y=f8,
        z=m8,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#FF1493',
        flatshading=True,
        name = "Loyal Customers"
    ),
    go.Mesh3d(
        x=r9,
        y=f9,
        z=m9,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6,
        color='#FF1F53',
        flatshading=True,
        name = "Champions"
    )
])

fig.update_layout(
    title="Distribución de segmentos RFM",
    scene=dict(
        xaxis=dict(title="Recency"),
        yaxis=dict(title="Frequency"),
        zaxis=dict(title="Monetary"),
    ),
    autosize=False,
    width=800,
    height=600,
    showlegend = True
)

#fig.show()

st.plotly_chart(fig)

# Generación del dicionario para guardar los grupos de segmentos
clusters_names = [
                  'lost',
                  'need_attention',
                  'past_customers',
                  'at_risk',
                  'new_customers',
                  'potential_loyaltist',
                  'low_spenders',
                  'loyal_customers',
                  'champions'
                  ]

cluster_definition = {}

for i,cluster in enumerate(clusters_names):
    # Define variables names
    r_start = f"r{i+1}_start"
    f_start = f"f{i+1}_start"
    m_start = f"m{i+1}_start"
    r_end = f"r{i+1}_end"
    f_end = f"f{i+1}_end"
    m_end = f"m{i+1}_end"

    # Edit dictionary accesing a varialbes based on the position in the list cluster
    cluster_definition.update(
        {cluster:{
            'recency': [f'G{x}' for x in list(range(locals()[r_start],locals()[r_end]+1))],
            'frequency': [f'G{x}' for x in list(range(locals()[f_start],locals()[f_end]+1))],
            'monetary': [f'G{x}' for x in list(range(locals()[m_start],locals()[m_end]+1))]
        }}
    )



# Escribimos en firebase los filtros elegidos y las variables seleccionadas, luego llamamos a la CF2 para que ejecute la query y arme los clusters

data = {
    "monetary":var_m,
    "frequency": var_f,
    "recency": var_r,
    #"split_users_with_one_transaction": str(remove_users_with_one_transaction),
    #"filter_remove_users" : str(remove_users),
    #"filter_variable" : when_variable,
    #"filter_condition" : is_,
    #"filter_value" : value,
    "n_buckets": 5,
    "cluster_definition" : cluster_definition

}


print(df.head())
