from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('interview/', views.interview, name='interview'),
    path('result/', views.result, name='result'),
    path('download-transcript/', views.download_transcript, name='download_transcript'),
    path('start-new/', views.start_new_assessment, name='start_new_assessment'),
]