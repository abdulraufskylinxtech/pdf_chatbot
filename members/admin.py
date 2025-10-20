from django.contrib import admin
from .models import CustomUser, UploadedFile, ChatbotQA
from django.contrib.auth.admin import UserAdmin

class CustomUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (
        ('Custom Fields', {'fields': ('is_main',)}),
    )
    list_display = ['username', 'email', 'is_main', 'is_staff']

admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(UploadedFile)
admin.site.register(ChatbotQA)
