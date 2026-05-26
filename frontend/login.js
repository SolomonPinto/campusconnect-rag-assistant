const loginForm = document.getElementById("login-form");

if (localStorage.getItem("studentProfile")) {
  window.location.replace("/chat");
}

loginForm.addEventListener("submit", (event) => {
  event.preventDefault();

  const formData = new FormData(loginForm);
  const profile = {
    name: String(formData.get("name")).trim(),
    studentId: String(formData.get("studentId")).trim(),
    email: String(formData.get("email")).trim(),
  };

  if (!profile.name || !profile.studentId || !profile.email) {
    return;
  }

  localStorage.setItem("studentProfile", JSON.stringify(profile));
  const sessionId =
    window.crypto && crypto.randomUUID
      ? crypto.randomUUID()
      : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  localStorage.setItem("sessionId", sessionId);
  window.location.href = "/chat";
});
