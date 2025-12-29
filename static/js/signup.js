//  Get elements
const form = document.querySelector('form');
const username = document.querySelector('input[name="username"]');
const email = document.querySelector('input[name="email"]');
const password = document.querySelector('input[name="password1"]');
const confirm = document.querySelector('input[name="password2"]');
const submitBtn = form.querySelector('button[type="submit"]');

//  Create dynamic error + success message areas
function createMessageSpan(input) {
  let span = document.createElement('span');
  span.classList.add('error');
  span.style.display = 'none';
  input.insertAdjacentElement('afterend', span);
  return span;
}

const usernameError = createMessageSpan(username);
const emailError = createMessageSpan(email);
const passwordError = createMessageSpan(password);
const confirmError = createMessageSpan(confirm);

//  Success message box (top of form)
let successMsg = document.createElement('div');
successMsg.classList.add('success-box');
successMsg.style.display = 'none';
successMsg.style.color = 'green';
successMsg.style.fontWeight = '600';
successMsg.style.marginBottom = '10px';
successMsg.style.fontSize = '14px';
form.insertAdjacentElement('beforebegin', successMsg);

//  Regex Rules
const usernameRegex = /^[a-z][a-z0-9_]{5,}$/; 
const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const specialChars = /[_!@#$%^&*(),.?":{}|<>+\-=]/;
const digitRegex = /\d/;
const firstCapital = /^[A-Z]/;

//  Clear error for a specific field
function clearError(span) {
  span.style.display = 'none';
  span.textContent = '';
}

// Username validation
function validateUsername() {
  const val = username.value.trim();
  clearError(usernameError);
  
  if (!val) {
    usernameError.textContent = ' Username is required.';
    usernameError.style.display = 'block';
    return false;
  }

  // Must start with lowercase letter and have at least 6 characters
  if (!/^[a-z0-9_]{6,}$/.test(val)) {
    usernameError.textContent =
      ' Must start with lowercase and be 6+ characters.';
    usernameError.style.display = 'block';
    return false;
  }

  // Must contain at least one number
  if (!/\d/.test(val)) {
    usernameError.textContent =
      ' Username must include at least one number (0–9).';
    usernameError.style.display = 'block';
    return false;
  }

  return true;
}


// Email validation
function validateEmail() {
  const val = email.value.trim();
  clearError(emailError);
  if (!val) {
    emailError.textContent = '';
    emailError.style.display = 'block';
    return false;
  }
  if (!emailRegex.test(val)) {
    emailError.textContent = ' Invalid email';
    emailError.style.display = 'block';
    return false;
  }
  return true;
}

//  Password validation
function validatePassword() {
  const val = password.value;
  clearError(passwordError);
  if (val.length < 8) {
    passwordError.textContent = ' Must be at least 8 characters.';
    passwordError.style.display = 'block';
    return false;
  }
  if (!firstCapital.test(val)) {
    passwordError.textContent = ' Must start with a capital letter.';
    passwordError.style.display = 'block';
    return false;
  }
  if (!digitRegex.test(val)) {
    passwordError.textContent = ' Must contain at least one number.';
    passwordError.style.display = 'block';
    return false;
  }
  if (!specialChars.test(val)) {
    passwordError.textContent =
      ' Must include one special character.';
    passwordError.style.display = 'block';
    return false;
  }
  return true;
}

//  Confirm password validation
function validateConfirm() {
  clearError(confirmError);
  if (confirm.value !== password.value || confirm.value === '') {
    confirmError.textContent = ' Passwords do not match.';
    confirmError.style.display = 'block';
    return false;
  }
  return true;
}

//  Check all fields valid
function allValid() {
  return (
    validateUsername() &&
    validateEmail() &&
    validatePassword() &&
    validateConfirm()
  );
}

//  Live validation (runs while typing)
username.addEventListener('input', () => {
  validateUsername();
  submitBtn.disabled = !allValid();
});
email.addEventListener('input', () => {
  validateEmail();
  submitBtn.disabled = !allValid();
});
password.addEventListener('input', () => {
  validatePassword();
  validateConfirm();
  submitBtn.disabled = !allValid();
});
confirm.addEventListener('input', () => {
  validateConfirm();
  submitBtn.disabled = !allValid();
});

//  Form submit event
form.addEventListener('submit', (e) => {
  if (!allValid()) {
    e.preventDefault();
    return false;
  }

  //  Show success message
  e.preventDefault(); // remove this line when backend handles redirect
  successMsg.textContent = ' Signup Successful! Redirecting...';
  successMsg.style.display = 'block';

  submitBtn.textContent = 'Processing...';
  submitBtn.disabled = true;
  submitBtn.style.background = '#4caf50';

  setTimeout(() => {
    form.submit();
  }, 1500);
});

document.querySelectorAll('.toggle-password').forEach(icon => {
  const targetId = icon.getAttribute('data-target');
  const input = document.getElementById(targetId);

  icon.addEventListener('click', () => {
    const isPassword = input.type === 'password';
    input.type = isPassword ? 'text' : 'password';
    icon.classList.toggle('fa-eye');
    icon.classList.toggle('fa-eye-slash');
  });
});


