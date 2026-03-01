const sendBtn = document.getElementById("sendBtn");
const userInput = document.getElementById("userInput");
const messages = document.getElementById("messages");
const chatItems = document.querySelectorAll(".chat-item");

let chatHistory = {};

// later based on data
const classList = [
  "MTH 2340",
  "CSC 3320",
  "CSC 2621",
  "CSC 3210",
  "PHL 3102"
];

let currentChat = classList[0];

/* Load saved data */
chrome.storage.local.get(["chatHistory", "lastChat"], (data) => {

  chatHistory = data.chatHistory || {};

  classList.forEach(cls => {
    if (!chatHistory[cls]) chatHistory[cls] = [];
  });

  currentChat = data.lastChat || classList[0];

  highlightCurrentChat();
  renderMessages();
});

/* Save */
function saveData() {
  chrome.storage.local.set({
    chatHistory,
    lastChat: currentChat
  });
}

function highlightCurrentChat() {
  chatItems.forEach(item => {
    item.classList.toggle(
      "active",
      item.dataset.chat === currentChat
    );
  });
}

function renderMessages() {
  messages.innerHTML = "";
  chatHistory[currentChat].forEach(msg => {
    addMessage(msg.text, msg.sender, false);
  });
}

function addMessage(text, sender, save = true) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  msg.textContent = text;
  messages.appendChild(msg);
  messages.scrollTop = messages.scrollHeight;

  if (save) {
    chatHistory[currentChat].push({ text, sender });
    saveData();
  }
}

function showSpinner() {
  const container = document.createElement("div");
  container.classList.add("message", "bot");
  container.id = "spinnerMessage";

  const spinner = document.createElement("div");
  spinner.classList.add("spinner");

  container.appendChild(spinner);
  messages.appendChild(container);
  messages.scrollTop = messages.scrollHeight;
}

function removeSpinner() {
  const spinner = document.getElementById("spinnerMessage");
  if (spinner) spinner.remove();
}

function handleSend() {
  const text = userInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  userInput.value = "";

  showSpinner();

  setTimeout(() => {
    removeSpinner();
    addMessage(`Lorem ipsum response for ${currentChat}.`, "bot");
  }, 1000);
}

/* Switch chats */
chatItems.forEach(item => {
  item.addEventListener("click", () => {
    currentChat = item.dataset.chat;
    highlightCurrentChat();
    renderMessages();
    saveData();
  });
});

sendBtn.addEventListener("click", handleSend);

userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") handleSend();
});