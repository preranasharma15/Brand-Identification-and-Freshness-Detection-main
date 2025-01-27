from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('brand_classification', views.brand_classification_view, name='brand_classification'),
    path('freshness_calculator', views.freshness_calculator_view, name='freshness_calculator'),
    path('brand_classification_feed', views.brand_classification_feed, name = 'brand_classification_feed'),
    path('freshness_calculator_feed', views.freshness_calculator_feed, name = 'freshness_calculator_feed'),
    path('export-brand', views.export_brand_results, name='export_brand_results'),
    path('export-freshness', views.export_freshness_results, name='export_freshness_results'),
]
