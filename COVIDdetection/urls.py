from django.conf.urls import include,url
from django.urls import path,re_path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.index, name='index'),    
    path('upload/', views.uploadImage, name='uploadImage'),
    url(r'^team/$', views.Team, name='Team'),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)