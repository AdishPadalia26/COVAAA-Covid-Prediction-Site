from __future__ import print_function
from enum import auto
import pandas as pd    # Data Handling library
import numpy as np     # Numrical Handling Library
import matplotlib.pyplot as plt    #Data Visualization
import seaborn as sns              #Data Visualization
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.core.display import display, HTML
import plotly.express as px
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet.plot import plot_plotly, plot_components_plotly
from prophet import Prophet
import json
import ipywidgets as widgets

# loading data right from the source:
death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
country_df = pd.read_csv("C:\\Users\\Anmol\\Desktop\\HELLO_FLASK\\mini_project_covid\\mini_project\\COVAAA\\datasets\\cases_country.csv")
confirmed_df1 = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
#confirmed_df=confirmed_df1.dropna()
confirmed_df = pd.read_csv('C:\\Users\\Anmol\\Desktop\\HELLO_FLASK\\mini_project_covid\\mini_project\COVAAA\datasets\\time_series_covid19_confirmed_global.csv')
#print(confirmed_df.head())
max = recovered_df.max(axis=1,skipna=True,numeric_only=True)

# Read csv file for India
df = pd.read_csv("C:\\Users\\Anmol\\Desktop\\HELLO_FLASK\\mini_project_covid\\mini_project\\COVAAA\\datasets\\covid_cases.csv")



# Droping the Province/State column as it has a lot of missing values
#df.drop(["State/UnionTerritory"],1,inplace=True)

world=df.groupby(['State/UnionTerritory']).max().reset_index()
cases=[]
cases = (world["Confirmed"]).tolist()
world.drop(['Date','Cured','Deaths'], axis = 1,inplace = True)


df["Date"]=pd.to_datetime(df["Date"])
grouped_country=df.groupby(["State/UnionTerritory","Date"]).agg({"Confirmed":'sum',"Cured":'sum',"Deaths":'sum'})
grouped_country["Active Cases"]=grouped_country["Confirmed"]-grouped_country["Cured"]-grouped_country["Deaths"]
datewise=df.groupby(["Date"]).agg({"Confirmed":'sum',"Cured":'sum',"Deaths":'sum'})
datewise["Days Since"]=datewise.index-datewise.index.min()

# Index Page
def data():
    l = [(len(df["State/UnionTerritory"].unique())),datewise["Confirmed"].iloc[-1],datewise["Cured"].iloc[-1]
    ,datewise["Deaths"].iloc[-1],(datewise["Confirmed"].iloc[-1]-datewise["Cured"].iloc[-1]-datewise["Deaths"].iloc[-1]),
    np.round(datewise["Confirmed"].iloc[-1]/datewise.shape[0]),np.round(datewise["Cured"].iloc[-1]/datewise.shape[0]),np.round(datewise["Deaths"].iloc[-1]/datewise.shape[0]),
    np.round(datewise["Confirmed"].iloc[-1]/((datewise.shape[0])*24)),np.round(datewise["Cured"].iloc[-1]/((datewise.shape[0])*24)),
    np.round(datewise["Deaths"].iloc[-1]/((datewise.shape[0])*24)),datewise["Confirmed"].iloc[-1]-datewise["Confirmed"].iloc[-2],
    datewise["Cured"].iloc[-1]-datewise["Cured"].iloc[-2],datewise["Deaths"].iloc[-1]-datewise["Deaths"].iloc[-2]]
    return l


# renaming the df column names to lowercase
country_df.columns = map(str.lower, country_df.columns)
confirmed_df.columns = map(str.lower, confirmed_df.columns)
death_df.columns = map(str.lower, death_df.columns)
recovered_df.columns = map(str.lower, recovered_df.columns)

# changing province/state to state and country/region to country
confirmed_df = confirmed_df.rename(columns={'province/state': 'state', 'country/region': 'country'})
recovered_df = confirmed_df.rename(columns={'province/state': 'state', 'country/region': 'country'})
death_df = death_df.rename(columns={'province/state': 'state', 'country/region': 'country'})
country_df = country_df.rename(columns={'country_region': 'country'})
confirmed_df1 = confirmed_df1.rename(columns={'Country/Region': 'country'})

def example():
    return confirmed_df1.head()

# total number of confirmed, death and recovered cases
confirmed_total = int(country_df['confirmed'].sum())
deaths_total = int(country_df['deaths'].sum())
recovered_total = int(country_df['recovered'].sum())
active_total = int(confirmed_total - deaths_total - recovered_total)

# displaying the total stats
def box():
    l={}
    l["Confirmed"] = confirmed_total
    l["Death"] = deaths_total
    l["Recovered"] = recovered_total
    l["Active"] = active_total
    return l



# sorting the values by confirmed descending order
fig = go.FigureWidget( layout=go.Layout() )
def highlight_col(x):
    r = 'background-color: #4AE0DF'
    y = 'background-color: #E0CD67'
    g = 'background-color: #E051BA'
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1.iloc[:, 4] = y
    df1.iloc[:, 5] = r
    df1.iloc[:, 6] = g
    
    return df1

def show_latest_cases(n):
    n = int(n)
    return (country_df.sort_values('confirmed', ascending= False).head(n))

# Showing table tof top 10 countries
def e1():
    # parsing the DataFrame in json format.
    json_records = show_latest_cases('10').reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    return data 

# Get Box stats for country
def Extract(country):
    ls = []
    for i in range(len(country_df)):
        print(country_df['country'][i])
        if country.capitalize() == country_df['country'][i]:
            ls = [int(country_df['confirmed'][i]),int(country_df['deaths'][i]),int(country_df['recovered'][i])]
            break
    return ls


# Get map for country
def plot_cases_of_a_country(country):
    labels = ['confirmed', 'deaths']
    colors = ['blue', 'red']
    mode_size = [6, 8]
    line_size = [4, 5]
    
    df_list = [confirmed_df1, death_df]
    
    fig = go.Figure()
    country = country.capitalize()
    
    for i, df in enumerate(df_list):
        if country == 'World' or country == 'world':
            x_data = np.array(list(df.iloc[:, 20:].columns))
            y_data = np.sum(np.asarray(df.iloc[:,4:]),axis = 0)
            
        else:    
            x_data = np.array(list(df.iloc[:, 20:].columns))
            y_data = np.sum(np.asarray(df[df['country'] == country].iloc[:,20:]),axis = 0)
            
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers',
        name=labels[i],
        line=dict(color=colors[i], width=line_size[i]),
        connectgaps=True,
        text = "Total " + str(labels[i]) +": "+ str(y_data[-1])
        ))
    
    fig.update_layout(
        title="COVID 19 cases of " + country,
        xaxis_title='Date',
        yaxis_title='No. of Confirmed Cases',
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="lightgrey",
        width = 800,
        
    )
    
    fig.update_yaxes(type="linear")
    graph_div = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return graph_div


# Predict Cases of countries
def Country_Predict(country):
  country_world_df = confirmed_df1[confirmed_df1['country'].str.startswith(country)]
  # country_world_df.drop(['Sno'], axis = 1,inplace = True)
  country_world_df.drop(['country','Province/State','Lat','Long'], axis = 1,inplace = True)
  country_world_df=country_world_df.T.reset_index().T
  # print(country_world_df)
  c_df=country_world_df.T
  # print(c_df)
  c_df.columns = ['ds','y']
  m = Prophet(interval_width=0.95, daily_seasonality=True,yearly_seasonality=True)
  model = m.fit(c_df)
  future = m.make_future_dataframe(periods=100,freq='D')
  forecast = m.predict(future)
  fig = plot_plotly(m, forecast)
  country_predict = plotly.offline.plot(fig, auto_open = False, output_type="div")
  return country_predict


sorted_country_df = country_df.sort_values('confirmed', ascending= False)

# plotting the 10 worst hit countries
def bubble_chart(n):
    fig = px.scatter(sorted_country_df.head(n), x="country", y="confirmed", size="confirmed", color="country",
               hover_name="country", size_max=60)
    fig.update_layout(
    title=str(n) +" Worst hit countries",
    xaxis_title="Countries",
    yaxis_title="Confirmed Cases",
    width = 700
    )
    graph_div = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return graph_div

# Bar Graph for Confirmed Cases in world
def bar_confirmed():
    fig = px.bar(
        sorted_country_df.head(10),
        x = "country",
        y = "confirmed",
        title= "Top 10 worst affected countries", # the axis names
        color_discrete_sequence=["aqua"], 
        height=500,
        width=800
    )
    confirmed = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return confirmed

# Bar Graph for Deaths in World
def bar_deaths():
    fig = px.bar(
        sorted_country_df.head(10),
        x = "country",
        y = "deaths",
        title= "Deaths in Top 10 worst affected countries", # the axis names
        color_discrete_sequence=["teal"], 
        height=500,
        width=800
    )
    deaths = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return deaths


#Worst hit countries - Recovering cases Bar Graph
def bar_recovered():
    fig = px.bar(
        sorted_country_df.head(10),
        x = "country",
        y = "recovered",
        title= "Recovered cases at Top 10 worst affected countries", # the axis names
        color_discrete_sequence=["gold"], 
        height=500,
        width=800
    )
    recov = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return recov

# World Map of cases
def map():
    fig = px.choropleth(country_df, locations='country',
                        locationmode="country names", color='confirmed', scope="world",color_continuous_scale=[[0, 'rgb(240,240,240)'],
                        [0.01, 'rgb(13,136,198)'],
                        [0.20, 'rgb(191,247,202)'],
                        [0.5, 'rgb(4,145,32)'],
                        [1, 'rgb(225,126,128)']])
    
    world_map = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return world_map

# Active Cases in India Graph
def active_india():
    fig=px.bar(x=datewise.index,y=datewise["Confirmed"]-datewise["Cured"]-datewise["Deaths"])
    fig.update_layout(title="Distribution of Number of Active Cases",
                    xaxis_title="Date",yaxis_title="Number of Cases",)
    active = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return active

# Cured Cases in India
def cured_india():
    fig=px.bar(x=datewise.index,y=datewise["Cured"])
    fig.update_layout(title="Distribution of Number of Cured Cases",
                    xaxis_title="Date",yaxis_title="Number of Cases")
    cured = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return cured

# Map of India
def map_india():
    fig = px.choropleth(
        world.dropna(),
        geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
        featureidkey='properties.ST_NM',
        color="Confirmed",
        locations='State/UnionTerritory',
        color_continuous_scale='Reds',
        width=800)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    india_map = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return india_map


#All 3 cases in 1 graph
def all():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"],
                        mode='lines+markers',
                        name='Confirmed Cases'))
    fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Deaths"],
                        mode='lines+markers',
                        name='Death Cases'))
    fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Cured"],
                        mode='lines+markers',
                        name='Recovered Cases'))
    fig.update_layout(title="Growth of different types of cases",
                    xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
    all = plotly.offline.plot(fig, auto_open = False, output_type="div")
    return all

# Prediction graph for each state
def Predict(state):
  state_df = df[df['State/UnionTerritory'].str.startswith(state)]
  state_df.drop(['Sno'], axis = 1,inplace = True)
  state_df.drop(['State/UnionTerritory','Cured','Deaths'], axis = 1,inplace = True)
  state_df.columns = ['ds','y']
  m = Prophet(interval_width=0.95, daily_seasonality=True,yearly_seasonality=True)
  model = m.fit(state_df)
  future = m.make_future_dataframe(periods=100,freq='D')
  forecast = m.predict(future)
  fig = plot_plotly(m, forecast)
  graph = plotly.offline.plot(fig, auto_open = False, output_type="div")
  return graph

# Box data for each state
def Extract_state(state):
    state_df = df[df['State/UnionTerritory'].str.startswith(state.capitalize())]
    state_df.drop(['Sno'], axis = 1,inplace = True)
    ls = []
    if len(state_df) != 0:
        ls = [int(state_df['Confirmed'].max()),int(state_df['Deaths'].max()),int(state_df['Cured'].max())]
    return ls

# All 3 cases in graph for each state
def plot_state(state):
  fig=go.Figure()
  state_df = df[df['State/UnionTerritory'].str.startswith(state)]
  state_df.drop(['Sno'], axis = 1,inplace = True)
  state_df.drop(['State/UnionTerritory'], axis = 1,inplace = True)
  fig.add_trace(go.Scatter(x=state_df["Date"], y=state_df["Confirmed"],
                      mode='lines+markers',
                      name='Confirmed Cases'))
  fig.add_trace(go.Scatter(x=state_df["Date"], y=state_df["Deaths"],
                      mode='lines+markers',
                      name='Death Cases'))
  fig.add_trace(go.Scatter(x=state_df["Date"], y=state_df["Cured"],
                      mode='lines+markers',
                      name='Recovered Cases'))
  fig.update_layout(title="Growth of different types of cases",
                  xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
  graph1 = plotly.offline.plot(fig, auto_open = False, output_type="div")
  return graph1