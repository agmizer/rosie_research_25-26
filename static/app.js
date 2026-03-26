let after = 0;

window.onload = () => {
  addMessage("ai", "Hi, I'm Sal. How can I help?");
  poll();

  document.getElementById("send").onclick = send;

  document.getElementById("msg").addEventListener("keydown", e => {
    if (e.key === "Enter") send();
  });

  document.getElementById("folder-btn").onclick = toggleOverlay;
};

function toggleOverlay() {
  const overlay = document.getElementById("overlay");
  overlay.style.display = overlay.style.display === "flex" ? "none" : "flex";
}

function addMessage(role, text) {
  const chat = document.getElementById("chat");

  const row = document.createElement("div");
  row.className = "row " + (role === "user" ? "user" : "ai");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerText = text;

  row.appendChild(bubble);
  chat.appendChild(row);

  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const input = document.getElementById("msg");
  const text = input.value.trim();
  if (!text) return;

  addMessage("user", text);
  input.value = "";

  await fetch("/api/send", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({message: text})
  });
}

async function poll() {
  try {
    const res = await fetch(`/api/messages?after=${after}`);
    const msgs = await res.json();

    for (const msg of msgs) {
      addMessage(msg.role === "student" ? "user" : "ai", msg.content);
      after++;
    }
  } catch {}

  setTimeout(poll, 1000);
}

let RAGInitialLoadData = {
  "test1": ["testtt.pdf", "testttt.pdf"],
  "test2": ["testtt.pdf"],
  "test3": ["testt.pdf"]
};

let currentFolder = null;

function renderFileSystem() {
  const overlay = document.getElementById("overlay-content");

  overlay.innerHTML = `
    <div style="display:flex; width:100%; height:100%;">

      <!-- LEFT: folders -->
      <div style="width:35%; border-right:2px solid #002f6c; padding:10px;">
        <button onclick="createFolder()">+ Add Folder</button>
        <div id="folderList"></div>
      </div>

      <!-- RIGHT: files -->
      <div style="flex:1; padding:10px;">
        <div style="display:flex; justify-content:space-between;">
          <h3 id="folderTitle">Folders</h3>
          <button onclick="goBack()">⬅ Back</button>
          <button onclick="createFile()">+ Add File</button>
        </div>

        <div id="fileList"></div>
      </div>

    </div>
  `;

  drawFolders();
}

function drawFolders() {
  const list = document.getElementById("folderList");
  list.innerHTML = "";

  Object.keys(RAGInitialLoadData).forEach(folder => {
    const div = document.createElement("div");
    div.innerText = folder;
    div.style.padding = "8px";
    div.style.cursor = "pointer";
    div.style.borderBottom = "1px solid #ccc";

    div.onclick = () => {
      currentFolder = folder;
      drawFiles();
    };

    list.appendChild(div);
  });
}

function drawFiles() {
  const list = document.getElementById("fileList");
  const title = document.getElementById("folderTitle");

  if (!currentFolder) {
    title.innerText = "Folders";
    list.innerHTML = "";
    return;
  }

  title.innerText = currentFolder;
  list.innerHTML = "";

  RAGInitialLoadData[currentFolder].forEach(file => {
    const div = document.createElement("div");
    div.innerText = file;
    div.style.padding = "6px";
    div.style.borderBottom = "1px solid #eee";
    list.appendChild(div);
  });
}

function createFolder() {
  const name = prompt("Folder name:");
  if (!name) return;

  if (!RAGInitialLoadData[name]) {
    RAGInitialLoadData[name] = [];
  }

  drawFolders();
}

function createFile() {
  if (!currentFolder) return;

  const name = prompt("File name:");
  if (!name) return;

  RAGInitialLoadData[currentFolder].push(name);
  drawFiles();
}

function goBack() {
  currentFolder = null;
  renderFileSystem();
}

function toggleOverlay() {
  const overlay = document.getElementById("overlay");

  overlay.style.display =
    overlay.style.display === "flex" ? "none" : "flex";

  if (overlay.style.display === "flex") {
    renderFileSystem();
  }
}