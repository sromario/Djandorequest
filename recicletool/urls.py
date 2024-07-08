from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('detection/', include('detection.urls')),  # chamar URLs do aplicativo detection
    path('', include('detection.urls')),  # Define a rota raiz
]
