const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const messages = document.getElementById("messages");
const loading = document.getElementById("loading");
const suggestions = document.getElementById("suggestions");
const newChatButton = document.getElementById("new-chat-button");

function createSessionId() {
  if (window.crypto && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getSessionId() {
  const existing = localStorage.getItem("sessionId");
  if (existing) return existing;
  const newId = createSessionId();
  localStorage.setItem("sessionId", newId);
  return newId;
}

let sessionId = getSessionId();

function scrollToLatest() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(text, role, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  article.appendChild(bubble);
  messages.appendChild(article);

  if (role === "assistant" && sources.length) {
    const sourceLine = document.createElement("p");
    sourceLine.className = "sources";
    sourceLine.textContent = `Sources: ${sources
      .map((source) => `${source.title} (${source.score.toFixed(2)})`)
      .join(" | ")}`;
    messages.appendChild(sourceLine);
  }
  scrollToLatest();
}

function setBusy(isBusy) {
  input.disabled = isBusy;
  sendButton.disabled = isBusy;
  loading.classList.toggle("hidden", !isBusy);
  if (isBusy) scrollToLatest();
}

async function sendMessage(rawMessage) {
  const message = rawMessage.trim();
  if (!message) return;
  suggestions.classList.add("hidden");
  addMessage(message, "user");
  input.value = "";
  setBusy(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, message }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "The request could not be completed.");
    }
    addMessage(data.reply, "assistant", data.sources || []);
  } catch (error) {
    addMessage(`Sorry, I could not complete that request. ${error.message}`, "assistant");
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

suggestions.addEventListener("click", (event) => {
  if (event.target.tagName === "BUTTON") {
    sendMessage(event.target.textContent);
  }
});

newChatButton.addEventListener("click", async () => {
  await fetch(`/api/chat/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  sessionId = createSessionId();
  localStorage.setItem("sessionId", sessionId);
  messages.innerHTML = "";
  addMessage("New conversation started. What would you like to know?", "assistant");
  input.focus();
});
