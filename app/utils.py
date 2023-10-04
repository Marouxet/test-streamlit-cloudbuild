import plotly
import pandas as pd

# Funcion que hace el grafico
def create_eda_plot(df,variable):
  '''
  Variable: one of 'r','f','m' or None
  '''
  # Importo librerias
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots

  # Save the first column in a variable to inicializate the plots
  first_column = df.columns[1]

  # Create DataFrame with basic metrics using pandas describe

  info = pd.DataFrame()
  for column in df.set_index('user_id').columns:
    info = pd.concat([info,df[column].describe()], axis = 1)
  info = info.apply(round,args={2})

  info.rename(
      index={
          'count':'Number of users',
          'mean': 'Mean value',
          'std' : 'Standard Deviation',
          'min' : 'Min value',
          '25%' : 'First quartile (25%)',
          '75%' : 'Third quartile (75%)',
          '50%' : 'Median',
          'max' : 'Max value',
          },
      inplace = True)

  info = info.applymap(str)


  # Table with units (only for RFM query)

  if variable == 'r':
    units = ['Days' for x in range(df.shape[1])]
    #units = ["$","$","Transactions", "Transactions", "Days", "Days","No unit"]
  elif variable == 'm':
    units = ['$' for x in range(df.shape[1])]
  elif variable == 'f':
    units = ['Transactions', 'Days']
    #units = ['Transactions' for x in range(df.shape[1])]
  else:
    units = ['' for x in range(df.shape[1])]
  # Add units to info dataframe
  for i,column in enumerate(info.columns):
    info.iloc[0,i] = str(info.iloc[0,i]) +  ' Users'
    info.iloc[1:,i] = info.iloc[1:,i].apply(lambda x:str(x) + ' ' + units[i])

  # Colours based on R, F, M
  if variable == 'r':
    color = "lightblue"
  elif variable == 'm':
    color = "lightcoral"
  elif variable == 'f':
    color = "peru"
  else:
    color = 'green'

  # Color de fondo
  bg_color = "rgb(255,255,255)"

  # Titles
  subplots_titles = ('Boxplot and histogram', '','','Table with metrics')

  # Initialize figure with subplots
  fig = make_subplots(
      rows=4, cols=1,
      subplot_titles=subplots_titles,
      vertical_spacing = 0.1,
      column_widths=[1],
      row_heights=[0.2, 0.2,0.1,0.5],
      shared_xaxes=True,
      specs=[[{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "table"}]]
      )


  fig.add_trace(
      go.Box(x=df[first_column],
            marker_color = color,
            ),
      row = 1,
      col = 1
  )
  fig.add_trace(
      go.Histogram(x=df[first_column],
            marker_color = color,
            ),
      row = 2,
      col = 1
  )

  fig.add_trace(
      go.Table(
      header=dict(values=["Metric","Value"],
                  ),
      cells=dict(values=[info.index, info[first_column]],
                )),
      row = 4,
      col = 1
  )



  # Add dropdown
  buttonlist = []

  for i,col in enumerate(df.set_index('user_id').columns):
      buttonlist.append(
          dict(
              args = [dict(
                  x=[df[str(col)]],

                  header=dict(values=["Metric","Value"],
                  fill_color=color,
                  align='left',
                  row=1,
                  col =1
                  ),

                cells=dict(values=[info.index, info[str(col)]],
                fill_color="lightgray",
                align='left'
                ),

            )],
              label=str(col),
              method='update'
          )
        )



  # Personalización estética:
  fig.update_layout(
    height = 800,
    width = 400,
    showlegend = False,

    updatemenus=[
              go.layout.Updatemenu(
                  buttons=buttonlist,
                  direction="down",
                  pad={"r": 0, "t": 0},
                  showactive=True,
                  x=0.1,
                  xanchor="left",
                  y=1.1,
                  yanchor="top"
              ),
          ],

  annotations=[
          dict(text="Variable:", showarrow=False,
          x=0, y=1.05, yref="paper", align="left")
      ],



  xaxis2 = dict(
          title = f'{units[0]}',
          showticklabels=True,
          visible = True,

  ),

  xaxis2_title = 'Check unit in table below',

  yaxis1 = dict(
        visible = False,

  ),

  yaxis2 = dict(
        title = 'Users',
        showticklabels=True,
        visible = True,

  ),
  # Cambio color de fondo del gráfico y del espacio entre gráficos.
  paper_bgcolor = bg_color,
  plot_bgcolor = "rgb(255,255,255)"
  )

  return fig