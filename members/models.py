from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models
from django.conf import settings
from .utils import delete_faiss_index

class CustomUser(AbstractUser):
    """Custom user model with admin flag and fixed related_name conflicts."""

    is_main = models.BooleanField(default=False)

    groups = models.ManyToManyField(
        Group,
        related_name="customuser_groups",
        blank=True,
        help_text="The groups this user belongs to.",
    )

    user_permissions = models.ManyToManyField(
        Permission,
        related_name="customuser_permissions",
        blank=True,
        help_text="Specific permissions for this user.",
    )

    def __str__(self):
        return f"{self.username} ({'Admin' if self.is_main else 'User'})"


class UploadedFile(models.Model):
    original_name = models.CharField(max_length=255)
    category = models.CharField(max_length=100)
    file = models.FileField(upload_to="uploads/")
    extracted_text = models.TextField(blank=True, null=True)  # store extracted content
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return self.original_name


    def delete(self, *args, **kwargs):
        """Delete FAISS index when file is deleted."""
        try:
            
            if self.uploaded_by:
                delete_faiss_index(self.id, self.uploaded_by.id)
        except Exception as e:
            print(f"Warning: FAISS index not deleted -> {e}")
        super().delete(*args, **kwargs)


class ChatbotQA(models.Model):
    """Stores user Q&A chat history for the chatbot."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=False,
        related_name="chatbot_qas",
    )
    uploaded_file = models.ForeignKey(
        'UploadedFile',              # Use string to avoid circular import issues
        on_delete=models.SET_NULL,   # Keep Q&A even if PDF is deleted
        null=True,
        blank=True,
        related_name="chatbot_qas_files"
    )
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.question[:50]}"
