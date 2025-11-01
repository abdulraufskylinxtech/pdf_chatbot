
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name="home"),
    path('signup/', views.signup_view, name="signup"),
    path('login/', views.login_view, name="login"),
    path('logout/', views.logout_view, name="logout"),
    path("main/", views.main_view, name="main"),
    path('chatbot/', views.chatbot_view, name="chatbot"),
    path('check-session/', views.check_session, name="check_session"),
    path('api/login/', views.api_login, name='api_login'),
    path('api/logout/', views.api_logout, name='api_logout'),
    path('api/upload/', views.api_upload_file, name='api_upload_file'),
    path('api/build_index/', views.api_build_index, name='api_build_index'),
    path('api/chat/', views.api_chatbot, name='api_chatbot'),
    path('api/history/', views.api_history, name='api_history'),
]
