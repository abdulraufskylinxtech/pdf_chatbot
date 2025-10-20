
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
    # path('api/upload/', views.api_upload, name='api_upload'),
    # path('api/chat/', views.api_chat, name='api_chat'),
    # path('api/history/', views.api_history, name='api_history'),
]
