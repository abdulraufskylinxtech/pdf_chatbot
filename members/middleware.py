from django.shortcuts import redirect
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin

class LoginRequiredMiddleware:
    """
    Redirect unauthenticated users to login page
    if they try to access protected pages manually.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        #  Skip authentication check for API routes
        if request.path.startswith("/api/"):
            return self.get_response(request)

        #  URLs that do NOT require authentication
        allowed_urls = [
            reverse('login'),
            reverse('signup'),
            reverse('home'),
        ]

        #  Allow admin, static, media
        if (
            request.path.startswith('/admin/') or
            request.path.startswith('/static/') or
            request.path.startswith('/media/')
        ):
            return self.get_response(request)

        #  Redirect if user is not authenticated
        if not request.user.is_authenticated:
            if not any(request.path == url or request.path.startswith(url) for url in allowed_urls):
                return redirect('login')

        return self.get_response(request)


class NoCacheForLoggedOutUsersMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        # Disable caching for all pages
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response
