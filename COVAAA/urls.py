from . import views
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.index,name="index"),
    path('example',views.example,name="example"),
    path('world',views.world,name="world"),
    path('india',views.india,name="india"),
    path("about", views.about, name="about"),
    path("state/<str:state>", views.states, name="state"),
    path("country/<str:country>", views.predict, name="country"),
]