from django.urls import path
from . import views

app_name = 'selector'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_dataset, name='upload'),
    path('results/', views.results, name='results'),
]
