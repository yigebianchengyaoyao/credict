"""credict URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp import views
urlpatterns = [
    path('',views.upload,name='upload'),
    path('charts/',views.loan_data_view,name='chart_view'),
    path('charts1/',views.loan_data_view1,name='chart_view1'),
    path('charts2/',views.loan_data_view2,name='chart_view2'),
    path('charts3/',views.loan_data_view3,name='chart_view3'),
    path('charts4/',views.loan_data_view4,name='chart_view4'),
]
