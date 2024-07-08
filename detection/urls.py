from django.urls import path
from . import views

urlpatterns = [
    path('', views.detect_objects_view, name='detect_objects'),  # Define a rota raiz
    path('detection/', views.detect_objects_view, name='detect_objects'),  # Mantém a rota 
]
