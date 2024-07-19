from django.http import HttpResponse,HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import redirect, render
from django import forms
from .helpers import Extract_state,Predict,Country_Predict, bar_recovered, example,data,box,e1,bar_confirmed,bar_deaths,map,bubble_chart,cured_india,active_india,map_india,all,plot_cases_of_a_country,Extract, plot_state

# Create your views here.
def example(request):
    return render(request,"COVAAA/temp.html",{
        #"message":example().to_html,
        #"m":plot_cases_of_a_country("India"),
        "map":map()
    })

def index(request):
    lst = data()
    res_dct = {i: lst[i] for i in range(0, len(lst))}
    return render(request,"COVAAA/index.html",{
        "data":res_dct,
    })

def world(request):
    if request.method == "POST":
        country = request.POST['country']
        if not country:
            return HttpResponseRedirect(reverse(world))
        else:
            return HttpResponseRedirect(f"country/{country}")
    else:
        return render(request,"COVAAA/world.html",{
            "cases": box(),
            "world_map": map(),
            "table": e1(),
            "bar_confirmed": bar_confirmed(),
            "bar_deaths":bar_deaths(),
            "bar_recovered":bar_recovered(),
            "bubble":bubble_chart(10),
        })

def india(request):
    if request.method == "POST":
        state = request.POST['state_name']
        if not state:
            return HttpResponseRedirect(reverse(india))
        else:
            return HttpResponseRedirect(f"state/{state}")
    else:
        return render(request,"COVAAA/india.html",{
            "cured": cured_india(),
            "active": active_india(),
            "all": all(),
            "map": map_india(),
        })

def predict(request,country):
    ls = Extract(country)
    if len(ls) == 0:
        return HttpResponseRedirect(reverse(world))
    ls.append(ls[0] - ls[1] - ls[2])
    box = {i: int(ls[i]) for i in range(0, len(ls))}
    country = country.capitalize()
    return render(request,"COVAAA/state-wise.html",{
        "graph":plot_cases_of_a_country(country),
        "box": box,
        "name":country,
        "predict_graph": Country_Predict(country)
    })

def states(request,state):
    ls = Extract_state(state)
    if len(ls) == 0:
        return HttpResponseRedirect(reverse(india))
    ls.append(int(ls[0])-int(ls[1])-int(ls[2]))
    box = {i: int(ls[i]) for i in range(0, len(ls))}
    return render(request,"COVAAA/state-wise.html",{
        "predict_graph":Predict(state),
        "box": box,
        "name":state,
        "graph" : plot_state(state),
    })

def about(request):
    return render(request,"COVAAA/aboutus.html")