// Get form and elements
const loginForm = document.querySelector('form');
const usernameInput = loginForm.querySelector('input[name="username"]');
const passwordInput = loginForm.querySelector('input[name="password"]');

// Create error span below each input
function createErrorSpan(input) {
    let span = document.createElement('span');
    span.classList.add('error');
    span.style.display = 'none';
    span.style.color = 'red';
    span.style.fontSize = '13px';
    span.style.marginTop = '5px';
    input.parentNode.insertBefore(span, input.nextSibling);
    return span;
}

// Create success message box (top of form)
let successMsg = document.createElement('div');
successMsg.classList.add('success-msg');
successMsg.style.display = 'none';
successMsg.style.color = 'green';
successMsg.style.fontSize = '15px';
successMsg.style.marginBottom = '10px';
successMsg.style.transition = 'all 0.4s ease';
loginForm.insertBefore(successMsg, loginForm.firstChild);

const usernameError = createErrorSpan(usernameInput);
const passwordError = createErrorSpan(passwordInput);

// Regex patterns
const usernameRegex = /^[a-z][a-z0-9_]{3,}$/;
const passwordRegex = /^(?=[A-Z])(?=.*\d)(?=.*[+,@,!#$%^&*_=-]).{8,}$/;

// Helper functions
function showError(span, message) {
    span.textContent = message;
    span.style.display = 'block';
}

function clearError(span) {
    span.style.display = 'none';
    span.textContent = '';
}

// Validate username
function validateUsername() {
    const val = usernameInput.value.trim();
    clearError(usernameError);

    if (!val) {
        showError(usernameError, " Username is required");
        return false;
    }
    if (!usernameRegex.test(val)) {
        showError(usernameError, " Invalid username");
        return false;
    }
    return true;
}

// Validate password
function validatePassword() {
    const val = passwordInput.value;
    clearError(passwordError);

    if (!val) {
        showError(passwordError, " Password is required");
        return false;
    }
    if (!passwordRegex.test(val)) {
        showError(passwordError, " Invalid password");
        return false;
    }
    return true;
}

// Show success message
function showSuccess(message) {
    successMsg.textContent = message;
    successMsg.style.display = 'block';
    successMsg.style.opacity = '1';

    setTimeout(() => {
        successMsg.style.opacity = '0';
        successMsg.style.display = 'none';
    }, 1000); // Hide after 2.5 seconds
}

// Form submit handler
loginForm.addEventListener('submit', (e) => {
    const validUsername = validateUsername();
    const validPassword = validatePassword();

    if (!(validUsername && validPassword)) {
        e.preventDefault(); // stop form if invalid
        return;
    }

    //  If all validations pass
    showSuccess("");
});

// Validate on blur (when leaving input)
usernameInput.addEventListener('blur', validateUsername);
passwordInput.addEventListener('blur', validatePassword);
