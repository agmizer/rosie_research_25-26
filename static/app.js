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

  // re render
  if (window.MathJax) {
    MathJax.typesetPromise([bubble]);
  }
}

async function send() {
  const input = document.getElementById("msg");
  const text = input.value.trim();
  if (!text) return;

  addMessage("user", text);
  input.value = "";

  await fetch("/api/send", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text })
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

let RAGInitialLoadData = {};
let currentFolder = null;

async function fetchFileSystem() {
  const res = await fetch("/api/files");
  RAGInitialLoadData = await res.json();
}

function toggleOverlay() {
  const overlay = document.getElementById("overlay");
  const open = overlay.style.display === "flex";

  if (!open) {
    fetchFileSystem().then(() => {
      overlay.style.display = "flex";
      currentFolder = null;
      renderFileSystem();
    });
  } else {
    closeOverlay();
  }
}

function closeOverlay() {
  document.getElementById("overlay").style.display = "none";
  currentFolder = null;
}


function renderFileSystem() {
  const root = document.getElementById("overlay-content");

  root.innerHTML = `
    <div style="
      position:relative; width:100%; height:100%;
      display:flex; flex-direction:column; background:white;
    ">
      <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding:10px; border-bottom:2px solid #C12033;
      ">
        <button onclick="closeOverlay()" style="
          background:#C12033; color:white; border:none;
          width:32px; height:32px; border-radius:6px;
          font-weight:bold; cursor:pointer;
        ">✖</button>
        <div id="topControls"></div>
      </div>
      <div id="fileSystemBody" style="flex:1; overflow:auto; padding:20px;"></div>
    </div>
  `;

  renderView();
}

function renderView() {
  if (!currentFolder) renderFolders();
  else renderFiles();
}

function renderFolders() {
  document.getElementById("topControls").innerHTML = `
    <button onclick="createFolder()" style="
      width:32px; height:32px; border-radius:6px; border:none;
      background:#C12033; color:white; font-size:20px; cursor:pointer;
    ">+</button>
  `;

  const body = document.getElementById("fileSystemBody");
  const folders = Object.keys(RAGInitialLoadData).filter(k => k !== "__root__");
  const rootFiles = RAGInitialLoadData["__root__"] || [];

  if (folders.length === 0 && rootFiles.length === 0) {
    body.innerHTML = `<p style="color:#999; text-align:center;">No files or folders yet. Press + to create a folder.</p>`;
    return;
  }

  const foldersHTML = folders.length === 0 ? "" : `
    <div style="
      display:grid;
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
      gap:20px;
      margin-bottom:24px;
    ">
      ${folders.map(folder => `
        <div style="
          display:flex; flex-direction:column; align-items:center;
          padding:10px; border-radius:10px; position:relative;
        ">
          <div onclick="openFolder('${folder}')" style="
            width:60px; height:60px; background:#C12033; border-radius:10px;
            display:flex; align-items:center; justify-content:center;
            color:white; font-size:26px; cursor:pointer;
          ">📁</div>
          <div style="margin-top:8px; font-size:14px; text-align:center;">${folder}</div>
          <button onclick="deleteFolder('${folder}')" style="
            position:absolute; top:4px; right:4px;
            background:transparent; border:none; color:#aaa;
            font-size:14px; cursor:pointer; line-height:1;
          " title="Delete folder">✕</button>
        </div>
      `).join("")}
    </div>
  `;

  const rootFilesHTML = rootFiles.length === 0 ? "" : `
    <div>
      <div style="font-size:13px; color:#999; margin-bottom:10px; text-transform:uppercase; letter-spacing:0.05em;">
        Loose files
      </div>
      <div style="display:flex; flex-direction:column; gap:10px;">
        ${rootFiles.map(file => `
          <div style="
            padding:10px; border:1px solid #eee; border-radius:8px;
            display:flex; justify-content:space-between; align-items:center;
          ">
            <span>📄 ${file}</span>
            <button onclick="deleteRootFile('${file}')" style="
              background:transparent; border:none; color:#aaa;
              font-size:16px; cursor:pointer;
            " title="Delete file">✕</button>
          </div>
        `).join("")}
      </div>
    </div>
  `;

  body.innerHTML = foldersHTML + rootFilesHTML;
}

function renderFiles() {
  document.getElementById("topControls").innerHTML = `
    <div style="display:flex; gap:10px; align-items:center;">
      <button onclick="goBack()" style="
        background:#C12033; color:white; border:none;
        width:32px; height:32px; border-radius:6px;
        cursor:pointer; font-size:18px;
      ">←</button>
      <button onclick="createFile()" style="
        width:32px; height:32px; border-radius:6px; border:none;
        background:#C12033; color:white; font-size:20px; cursor:pointer;
      ">+</button>
    </div>
  `;

  const body = document.getElementById("fileSystemBody");
  const files = RAGInitialLoadData[currentFolder] || [];

  if (files.length === 0) {
    body.innerHTML = `<p style="color:#999; text-align:center;">No files yet. Press + to add one.</p>`;
    return;
  }

  body.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:10px;">
      ${files.map(file => `
        <div style="
          padding:10px; border:1px solid #eee; border-radius:8px;
          display:flex; justify-content:space-between; align-items:center;
        ">
          <span>📄 ${file}</span>
          <button onclick="deleteFile('${file}')" style="
            background:transparent; border:none; color:#aaa;
            font-size:16px; cursor:pointer;
          " title="Delete file">✕</button>
        </div>
      `).join("")}
    </div>
  `;
}

function openFolder(name) {
  currentFolder = name;
  renderView();
}

function goBack() {
  currentFolder = null;
  renderView();
}

async function createFolder() {
  const name = prompt("Folder name:");
  if (!name || !name.trim()) return;

  const res = await fetch("/api/files/folder", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder: name.trim() })
  });

  if (res.ok) {
    RAGInitialLoadData[name.trim()] = [];
    renderView();
  } else {
    const err = await res.json();
    alert("Error: " + err.error);
  }
}

async function createFile() {
  const name = prompt("File name:");
  if (!name || !name.trim()) return;

  const res = await fetch("/api/files/file", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder: currentFolder, filename: name.trim() })
  });

  if (res.ok) {
    RAGInitialLoadData[currentFolder].push(name.trim());
    renderView();
  } else {
    const err = await res.json();
    alert("Error: " + err.error);
  }
}

async function deleteFolder(name) {
  if (!confirm(`Delete folder "${name}" and all its files?`)) return;

  const res = await fetch(`/api/files/folder?folder=${encodeURIComponent(name)}`, {
    method: "DELETE"
  });

  if (res.ok) {
    delete RAGInitialLoadData[name];
    renderView();
  } else {
    const err = await res.json();
    alert("Error: " + err.error);
  }
}

async function deleteFile(filename) {
  if (!confirm(`Delete "${filename}"?`)) return;

  const res = await fetch(
    `/api/files/file?folder=${encodeURIComponent(currentFolder)}&filename=${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );

  if (res.ok) {
    RAGInitialLoadData[currentFolder] = RAGInitialLoadData[currentFolder].filter(f => f !== filename);
    renderView();
  } else {
    const err = await res.json();
    alert("Error: " + err.error);
  }
}

// Deletes a file sitting directly in RAGInitialLoadData (not inside a subfolder)
async function deleteRootFile(filename) {
  if (!confirm(`Delete "${filename}"?`)) return;

  const res = await fetch(
    `/api/files/file?folder=&filename=${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );

  if (res.ok) {
    RAGInitialLoadData["__root__"] = RAGInitialLoadData["__root__"].filter(f => f !== filename);
    renderView();
  } else {
    const err = await res.json();
    alert("Error: " + err.error);
  }
}