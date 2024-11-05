# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    # path('predict/cnn/', views.predict_cnn, name='predict_cnn'),
    # path('predict/dense/', views.predict_dense, name='predict_dense'),
    path('predict/lstm/', views.predict_lstm, name='predict_lstm'),
    path('visualizations/', views.visualizations, name='visualizations'),
    path('compare/', views.compare_predictions, name='compare_predictions'),
    path('download-report/', views.download_report, name='download_report'),
    path('real-time-prediction/', views.real_time_prediction, name='real_time_prediction'),
]
