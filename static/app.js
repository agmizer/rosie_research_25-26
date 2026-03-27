const BASE = window.location.pathname.replace(/\/+$/, "");
let after = 0;
let RAGInitialLoadData = {};
let openCourses = new Set();

window.onload = () => {
  poll();

  document.getElementById("send").onclick = send;
  document.getElementById("msg").addEventListener("keydown", e => {
    if (e.key === "Enter") send();
  });

  document.getElementById("sidebar-toggle").onclick = () => {
    document.getElementById("sidebar").classList.toggle("collapsed");
  };

  document.getElementById("add-course-btn").onclick = createCourse;

  // Load sidebar on start
  fetchFileSystem().then(renderSidebar);
};

// ===================== Markdown / Messages =====================

function markdownToHtml(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  const lines = escaped.split("\n");
  const result = [];
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    if (/^#{3,6}\s+/.test(line)) {
      const level = line.match(/^(#{3,6})/)[1].length;
      line = `<h${level}>${line.replace(/^#{3,6}\s+/, "")}</h${level}>`;
    } else if (/^##\s+/.test(line)) {
      line = `<h2>${line.replace(/^##\s+/, "")}</h2>`;
    } else if (/^#\s+/.test(line)) {
      line = `<h1>${line.replace(/^#\s+/, "")}</h1>`;
    }
    result.push(line);
  }

  return result.join("\n")
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/__(.+?)__/g, "<b>$1</b>");
}

function addMessage(role, text) {
  const chat = document.getElementById("chat");

  const row = document.createElement("div");
  row.className = "row " + (role === "user" ? "user" : "ai");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = markdownToHtml(text);

  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;

  if (window.MathJax) {
    MathJax.typesetPromise([bubble]);
  }
}

async function send() {
  const input = document.getElementById("msg");
  const text = input.value.trim();
  if (!text) return;

  const welcome = document.getElementById("welcome");
  if (welcome) welcome.style.display = "none";
  document.getElementById("chat").style.display = "flex";

  addMessage("user", text);
  input.value = "";

  await fetch(`${BASE}/api/send`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text })
  });
}

async function poll() {
  try {
    const res = await fetch(`${BASE}/api/messages?after=${after}`);
    const msgs = await res.json();
    for (const msg of msgs) {
      addMessage(msg.role === "student" ? "user" : "ai", msg.content);
      after++;
    }
  } catch {}
  setTimeout(poll, 1000);
}

// ===================== Sidebar / File System =====================

async function fetchFileSystem() {
  const res = await fetch(`${BASE}/api/files`);
  RAGInitialLoadData = await res.json();
}

function escAttr(s) {
  return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderSidebar() {
  const list = document.getElementById("sidebar-list");
  const courses = Object.keys(RAGInitialLoadData).filter(k => k !== "__root__");

  if (courses.length === 0) {
    list.innerHTML = `<div style="padding:16px; color:#999; font-size:13px; text-align:center;">No courses yet. Press + to add one.</div>`;
    return;
  }

  list.innerHTML = courses.map(course => {
    const files = RAGInitialLoadData[course] || [];
    const isOpen = openCourses.has(course);
    return `
      <div class="course">
        <div class="course-row" onclick="toggleCourse('${escAttr(course)}')">
          <span class="chevron ${isOpen ? "open" : ""}">&#9654;</span>
          <span class="course-name" title="${escAttr(course)}">${escHtml(course)}</span>
          <span class="course-actions" onclick="event.stopPropagation()">
            <button onclick="addFileToCourse('${escAttr(course)}')" title="Add file">+</button>
            <button onclick="deleteCourse('${escAttr(course)}')" title="Delete course">&times;</button>
          </span>
        </div>
        <div class="file-list ${isOpen ? "open" : ""}">
          ${files.map(f => `
            <div class="file-item">
              <span class="file-name" title="${escAttr(f)}">${escHtml(f)}</span>
              <button class="file-delete" onclick="deleteFileFromCourse('${escAttr(course)}','${escAttr(f)}')">&times;</button>
            </div>
          `).join("")}
          ${files.length === 0 ? `<div class="file-item" style="color:#bbb; font-style:italic;">No files</div>` : ""}
        </div>
      </div>
    `;
  }).join("");
}

function toggleCourse(name) {
  if (openCourses.has(name)) {
    openCourses.delete(name);
  } else {
    openCourses.add(name);
  }
  renderSidebar();
}

async function createCourse() {
  const name = prompt("Course name:");
  if (!name || !name.trim()) return;

  const res = await fetch(`${BASE}/api/files/folder`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder: name.trim() })
  });

  if (res.ok) {
    RAGInitialLoadData[name.trim()] = [];
    openCourses.add(name.trim());
    renderSidebar();
  } else {
    const err = await res.json();
    alert("Error: " + err.error);
  }
}

async function addFileToCourse(course) {
  const name = prompt(`Add file to "${course}":`);
  if (!name || !name.trim()) return;

  const res = await fetch(`${BASE}/api/files/file`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder: course, filename: name.trim() })
  });

  if (res.ok) {
    RAGInitialLoadData[course].push(name.trim());
    openCourses.add(course);
    renderSidebar();
  } else {
    const err = await res.json();
    alert("Error: " + err.error);
  }
}

async function deleteCourse(name) {
  if (!confirm(`Delete "${name}" and all its files?`)) return;

  const res = await fetch(`${BASE}/api/files/folder?folder=${encodeURIComponent(name)}`, {
    method: "DELETE"
  });

  if (res.ok) {
    delete RAGInitialLoadData[name];
    openCourses.delete(name);
    renderSidebar();
  } else {
    const err = await res.json();
    alert("Error: " + err.error);
  }
}

async function deleteFileFromCourse(course, filename) {
  if (!confirm(`Delete "${filename}"?`)) return;

  const res = await fetch(
    `${BASE}/api/files/file?folder=${encodeURIComponent(course)}&filename=${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );

  if (res.ok) {
    RAGInitialLoadData[course] = RAGInitialLoadData[course].filter(f => f !== filename);
    renderSidebar();
  } else {
    const err = await res.json();
    alert("Error: " + err.error);
  }
}
