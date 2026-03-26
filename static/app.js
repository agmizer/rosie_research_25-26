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

function toggleOverlay() {
  const overlay = document.getElementById("overlay");
  const open = overlay.style.display === "flex";

  overlay.style.display = open ? "none" : "flex";

  if (!open) renderFileSystem();
}

function closeOverlay() {
  document.getElementById("overlay").style.display = "none";
  currentFolder = null;
}

//main
function renderFileSystem() {
  const root = document.getElementById("overlay-content");

  root.innerHTML = `
    <div style="
      position:relative;
      width:100%;
      height:100%;
      display:flex;
      flex-direction:column;
      background:white;
    ">

      <!-- TOP BAR -->
      <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding:10px;
        border-bottom:2px solid #C12033;
      ">

        <!-- CLOSE (TOP LEFT X) -->
        <button onclick="closeOverlay()"
          style="
            background:#C12033;
            color:white;
            border:none;
            width:32px;
            height:32px;
            border-radius:6px;
            font-weight:bold;
            cursor:pointer;
          ">
          ✖
        </button>

        <div id="topControls"></div>

      </div>

      <!-- BODY -->
      <div id="fileSystemBody" style="
        flex:1;
        overflow:auto;
        padding:20px;
      "></div>

    </div>
  `;

  renderView();
}

//switch
function renderView() {
  if (!currentFolder) renderFolders();
  else renderFiles();
}

function renderFolders() {
  const body = document.getElementById("fileSystemBody");
  const top = document.getElementById("topControls");

  top.innerHTML = ""; // no back button in root

  body.innerHTML = `
    <div style="
      display:grid;
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
      gap:20px;
    ">
      ${Object.keys(RAGInitialLoadData).map(folder => `
        <div onclick="openFolder('${folder}')"
          style="
            display:flex;
            flex-direction:column;
            align-items:center;
            cursor:pointer;
            padding:10px;
            border-radius:10px;
          "
        >
          <div style="
            width:60px;
            height:60px;
            background:#C12033;
            border-radius:10px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:white;
            font-size:26px;
          ">
            📁
          </div>
          <div style="margin-top:8px; font-size:14px;">
            ${folder}
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

// file system view
function renderFiles() {
  const body = document.getElementById("fileSystemBody");
  const top = document.getElementById("topControls");

  top.innerHTML = `
  <div style="display:flex; gap:10px; align-items:center;">

    <button onclick="goBack()"
      style="
        background:#C12033;
        color:white;
        border:none;
        width:32px;
        height:32px;
        border-radius:6px;
        cursor:pointer;
        font-size:18px;
      ">
      ←
    </button>

    <button onclick="createFile()"
      style="
        margin-left:10px;
        width:32px;
        height:32px;
        border-radius:6px;
        border:none;
        background:#C12033;
        color:white;
        font-size:20px;
        cursor:pointer;
        display:flex;
        align-items:center;
        justify-content:center;
      ">
      +
    </button>

  </div>
`;

  body.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:10px;">
      ${(RAGInitialLoadData[currentFolder] || []).map(file => `
        <div style="
          padding:10px;
          border:1px solid #eee;
          border-radius:8px;
        ">
          📄 ${file}
        </div>
      `).join("")}
    </div>
  `;
}

//actions
function openFolder(name) {
  currentFolder = name;
  renderView();
}

function goBack() {
  currentFolder = null;
  renderView();
}

function createFile() {
  const name = prompt("File name:");
  if (!name) return;

  RAGInitialLoadData[currentFolder].push(name);
  renderView();
}