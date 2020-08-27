"""
Definition of urls for DataScienceProject.
"""

from datetime import datetime
from django.urls import include, path
from COVIDdetection import forms, views
from django.conf import settings
from django.conf.urls import include,url
from django.conf.urls.static import static

urlpatterns = [    

    #path(r'COVID/', include('COVIDdetection.urls')),
    #path('upload/', views.uploadImage, name='uploadImage'),
    url(r'^$', views.index, name='index'),    
    url(r'^upload/$', views.uploadImage, name='uploadImage'),
    url(r'^team/$', views.team, name='team'),
    #url(r'^patientList/(?P<value>\d+)/$', views.patientList, name='patientList'),
    path(r'^patientList/<str:patient_name>/$',views.patientList, name='patientList'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
