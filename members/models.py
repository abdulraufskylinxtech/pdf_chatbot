from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models
from django.utils import timezone
from django.conf import settings
from .utils import delete_faiss_index
import uuid

class CustomUser(AbstractUser):
    """Custom user model with admin flag and fixed related_name conflicts."""

    is_main = models.BooleanField(default=False)
    session_id = models.CharField(max_length=100, blank=True, null=True, help_text="Current active session ID")

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
    extracted_text = models.TextField(blank=True, null=True) 
    file_hash = models.CharField(max_length=64, blank=True, null=True)
    session_id = models.CharField(max_length=100, db_index=True, default="")
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


class FileChunk(models.Model):
    file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.IntegerField()  
    chunk_text = models.TextField()       
    embedding = models.BinaryField()      
    index_path = models.CharField(max_length=500)
    total_chunks = models.IntegerField()  
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Chunk {self.chunk_index} of File {self.file.id}"



class ChatbotQA(models.Model):
    """Stores user Q&A chat history for the chatbot."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=False,
        blank=False,
        related_name="chatbot_qas",
    )
    uploaded_file = models.ForeignKey(
        'UploadedFile',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="qa_from_file"
    )

    username = models.CharField(max_length=150, db_index=True)
    session_id = models.CharField(max_length=100, db_index=True)

    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Chatbot QA"
        verbose_name_plural = "Chatbot QAs"

    def __str__(self):
        file_info = f" (File ID: {self.uploaded_file.id})" if self.uploaded_file else ""
        return f"{self.user.username}{file_info} → {self.question[:50]}"




class ApiUser(AbstractUser):
    """
    Simplified custom user model for API login.
    Uses username + email for authentication.
    """
    email = models.EmailField(unique=True)
    session_id = models.CharField(max_length=100, blank=True, null=True)
    is_main = models.BooleanField(default=False)  

    def __str__(self):
        return f"{self.username} ({self.email})"

    def generate_session(self):
        """Generate and save a new session ID"""
        self.session_id = uuid.uuid4().hex
        self.save(update_fields=['session_id'])
        return self.session_id


class ApiUploadedFile(models.Model):  
    uploaded_by = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    file = models.FileField(upload_to="uploads/")
    original_name = models.CharField(max_length=255)
    category = models.CharField(max_length=100, default="General")
    extracted_text = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)


class ApiChatMessage(models.Model):
    user = models.ForeignKey(ApiUser, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100, db_index=True)
    question = models.TextField()
    answer = models.TextField(blank=True, null=True)
    references = models.JSONField(null=True, blank=True) 
    created_at = models.DateTimeField(auto_now_add=True)



class InterviewSession(models.Model):
    STATUS_CHOICES = (
        ("pending", "Pending"),
        ("in_progress", "In Progress"),
        ("completed", "Completed"),
        ("cancelled", "Cancelled"),
    )
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="interview_sessions")
    role = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    cv_text = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    total_score = models.FloatField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)  

    def mark_started(self):
        self.started_at = timezone.now()
        self.status = "in_progress"
        self.save(update_fields=["started_at","status"])

    def mark_completed(self):
        self.completed_at = timezone.now()
        self.status = "completed"
        self.save(update_fields=["completed_at","status"])

    def __str__(self):
        return f"Interview {self.id} ({self.role}) - {self.user.username}"


class InterviewQuestion(models.Model):
    session = models.ForeignKey(InterviewSession, on_delete=models.CASCADE, related_name="questions")
    order = models.PositiveIntegerField(default=0)
    text = models.TextField()
    expected = models.TextField(blank=True, default="")  
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order"]

    def __str__(self):
        return f"Q{self.order} ({self.session.id})"


class InterviewResponse(models.Model):
    question = models.ForeignKey(InterviewQuestion, on_delete=models.CASCADE, related_name="responses")
    answer_text = models.TextField()
    score = models.FloatField(null=True, blank=True)
    feedback = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Response for Q{self.question.order} (score={self.score})"

