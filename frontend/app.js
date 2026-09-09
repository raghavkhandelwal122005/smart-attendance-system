const API = window.API_URL || localStorage.getItem("API_URL") || ""; // Same origin by default, or configurable backend URL

// State variables
let regPhotos = [];
let currentSessionId = null;
let currentClassName = "";
let capturedImageFile = null;
let allStudentsCache = [];
let lastRecognitionResults = [];
let lastRecognitionMeta = null;
let focusedRecordId = null;

// ---------------------------------------------------------------------
// Initialization & Navigation
// ---------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  loadStats();
  setupTabs();
  setupDragAndDrop();
  setupRegistrationForm();
  setupAttendanceWorkflow();
  setupStudentSearch();
  setupDemoSeeding();
  setupFocusReset();

  document.getElementById("att-date").valueAsDate = new Date();
});

function setupFocusReset() {
  const btn = document.getElementById("btn-show-all-faces");
  if (!btn) return;
  btn.addEventListener("click", () => {
    focusedRecordId = null;
    document.querySelectorAll(".result-card-item").forEach((item) => item.classList.remove("focused"));
    drawBoundingBoxes(lastRecognitionResults);
  });
}

function setupDemoSeeding() {
  const btn = document.getElementById("btn-seed-demo");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Loading Demo Data...";
    try {
      const res = await fetch(`${API}/api/demo/seed`, { method: "POST" });
      const data = await res.json();
      showToast(`Loaded demo data! Student Gwen (ID: 5) & Class gallery ready.`, "success");
      loadStats();
      loadStudents();
    } catch (err) {
      showToast("Failed to load demo data: " + err.message, "error");
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg> Load Demo Data`;
    }
  });
}

// Toast notification helper
function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 4000);
}

// Stats Loader
async function loadStats() {
  try {
    const res = await fetch(`${API}/api/stats`);
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById("stat-students").textContent = data.total_students || 0;
    document.getElementById("stat-photos").textContent = data.total_photos || 0;
    document.getElementById("stat-sessions").textContent = data.total_sessions || 0;
    document.getElementById("stat-present").textContent = data.total_present || 0;
  } catch (err) {
    console.error("Stats load error:", err);
  }
}

// Navigation Tabs
function setupTabs() {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add("active");

      if (btn.dataset.tab === "students") loadStudents();
      if (btn.dataset.tab === "records") loadSessions();
    });
  });
}

// Helper: Convert data URL to File
function dataURLtoFile(dataUrl, filename) {
  const arr = dataUrl.split(",");
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) u8arr[n] = bstr.charCodeAt(n);
  return new File([u8arr], filename, { type: mime });
}

async function stopStream(video) {
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
}

// ---------------------------------------------------------------------
// Drag and Drop File Handlers
// ---------------------------------------------------------------------
function setupDragAndDrop() {
  const regDropzone = document.getElementById("reg-dropzone");
  const regInput = document.getElementById("reg-file-input");

  regDropzone.addEventListener("click", () => regInput.click());
  regDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    regDropzone.classList.add("dragover");
  });
  regDropzone.addEventListener("dragleave", () => regDropzone.classList.remove("dragover"));
  regDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    regDropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      addRegPhotos(Array.from(e.dataTransfer.files));
    }
  });

  regInput.addEventListener("change", () => {
    if (regInput.files.length) addRegPhotos(Array.from(regInput.files));
    regInput.value = "";
  });

  const attDropzone = document.getElementById("att-dropzone");
  const attInput = document.getElementById("att-file-input");

  attDropzone.addEventListener("click", () => attInput.click());
  attDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    attDropzone.classList.add("dragover");
  });
  attDropzone.addEventListener("dragleave", () => attDropzone.classList.remove("dragover"));
  attDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    attDropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      setCapturedImage(e.dataTransfer.files[0]);
    }
  });

  attInput.addEventListener("change", () => {
    if (attInput.files[0]) setCapturedImage(attInput.files[0]);
    attInput.value = "";
  });
}

// ---------------------------------------------------------------------
// Register Student Workflow
// ---------------------------------------------------------------------
function addRegPhotos(files) {
  regPhotos.push(...files);
  renderRegThumbs();
}

function renderRegThumbs() {
  const thumbsGrid = document.getElementById("reg-thumbs");
  thumbsGrid.innerHTML = "";
  regPhotos.forEach((file, i) => {
    const item = document.createElement("div");
    item.className = "thumb-item";

    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);

    const removeBtn = document.createElement("button");
    removeBtn.className = "thumb-remove";
    removeBtn.innerHTML = "&times;";
    removeBtn.title = "Remove photo";
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      regPhotos.splice(i, 1);
      renderRegThumbs();
    });

    item.appendChild(img);
    item.appendChild(removeBtn);
    thumbsGrid.appendChild(item);
  });
}

function setupRegistrationForm() {
  const regWebcamBtn = document.getElementById("reg-webcam-btn");
  const regWebcamBox = document.getElementById("reg-webcam-box");
  const regVideo = document.getElementById("reg-video");
  const regCaptureBtn = document.getElementById("reg-capture-btn");
  const regWebcamClose = document.getElementById("reg-webcam-close");

  regWebcamBtn.addEventListener("click", async () => {
    regWebcamBox.classList.remove("hidden");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      regVideo.srcObject = stream;
    } catch (e) {
      showToast("Could not access webcam: " + e.message, "error");
      regWebcamBox.classList.add("hidden");
    }
  });

  regCaptureBtn.addEventListener("click", () => {
    const canvas = document.createElement("canvas");
    canvas.width = regVideo.videoWidth || 640;
    canvas.height = regVideo.videoHeight || 480;
    canvas.getContext("2d").drawImage(regVideo, 0, 0);
    const file = dataURLtoFile(canvas.toDataURL("image/jpeg", 0.92), `snapshot_${Date.now()}.jpg`);
    addRegPhotos([file]);
    showToast("Snapshot captured!", "success");
  });

  regWebcamClose.addEventListener("click", () => {
    stopStream(regVideo);
    regWebcamBox.classList.add("hidden");
  });

  document.getElementById("registerForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const submitBtn = document.getElementById("reg-submit-btn");
    submitBtn.disabled = true;
    submitBtn.textContent = "Saving Student Profile...";

    const studentId = document.getElementById("reg-student-id").value.trim();
    const name = document.getElementById("reg-name").value.trim();
    const className = document.getElementById("reg-class").value.trim();

    if (regPhotos.length === 0) {
      showToast("Please add at least one face reference photo.", "error");
      submitBtn.disabled = false;
      submitBtn.textContent = "Save & Register Student Profile";
      return;
    }

    const form = new FormData();
    form.append("student_id", studentId);
    form.append("name", name);
    form.append("class_name", className);
    regPhotos.forEach((f) => form.append("photos", f));

    try {
      const res = await fetch(`${API}/api/students`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Registration failed");

      showToast(`Registered ${data.name} (${data.student_id}) with ${data.photos_saved} face photo(s)!`, "success");
      document.getElementById("registerForm").reset();
      regPhotos = [];
      renderRegThumbs();
      stopStream(regVideo);
      regWebcamBox.classList.add("hidden");
      loadStats();
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Save & Register Student Profile";
    }
  });
}

// ---------------------------------------------------------------------
// Attendance Workflow & Inspector Sync
// ---------------------------------------------------------------------
function setupAttendanceWorkflow() {
  const attThreshold = document.getElementById("att-threshold");
  attThreshold.addEventListener("input", () => {
    document.getElementById("threshold-val").textContent = attThreshold.value;
  });

  document.getElementById("att-start-session").addEventListener("click", async () => {
    const className = document.getElementById("att-class").value.trim();
    const sessionDate = document.getElementById("att-date").value;
    if (!className) {
      showToast("Please enter a Class / Section name.", "error");
      return;
    }

    try {
      const res = await fetch(`${API}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class_name: className, session_date: sessionDate }),
      });
      const data = await res.json();
      currentSessionId = data.id;
      currentClassName = data.class_name;

      const banner = document.getElementById("att-session-banner");
      banner.classList.remove("hidden");
      document.getElementById("session-banner-text").textContent =
        `Active Session #${data.id} for "${data.class_name}" on ${data.session_date}`;

      document.getElementById("att-capture-area").classList.remove("hidden");
      document.getElementById("att-photo-summary").classList.add("hidden");
      document.getElementById("att-inspector").classList.add("hidden");
      document.getElementById("att-finalize-row").classList.add("hidden");

      allStudentsCache = await (await fetch(`${API}/api/students`)).json();
      showToast(`Session #${data.id} started successfully. Now upload classroom photo.`, "success");
      loadStats();
    } catch (err) {
      showToast("Failed to start session: " + err.message, "error");
    }
  });

  // Webcam Controls
  const attWebcamBtn = document.getElementById("att-webcam-btn");
  const attWebcamBox = document.getElementById("att-webcam-box");
  const attVideo = document.getElementById("att-video");
  const attCaptureBtn = document.getElementById("att-capture-btn");
  const attWebcamClose = document.getElementById("att-webcam-close");

  attWebcamBtn.addEventListener("click", async () => {
    attWebcamBox.classList.remove("hidden");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      attVideo.srcObject = stream;
    } catch (e) {
      showToast("Could not access webcam: " + e.message, "error");
      attWebcamBox.classList.add("hidden");
    }
  });

  attCaptureBtn.addEventListener("click", () => {
    const canvas = document.createElement("canvas");
    canvas.width = attVideo.videoWidth || 1280;
    canvas.height = attVideo.videoHeight || 720;
    canvas.getContext("2d").drawImage(attVideo, 0, 0);
    const file = dataURLtoFile(canvas.toDataURL("image/jpeg", 0.92), `classroom_${Date.now()}.jpg`);
    setCapturedImage(file);
    stopStream(attVideo);
    attWebcamBox.classList.add("hidden");
  });

  attWebcamClose.addEventListener("click", () => {
    stopStream(attVideo);
    attWebcamBox.classList.add("hidden");
  });

  const attDemoPhotoBtn = document.getElementById("att-demo-photo-btn");
  if (attDemoPhotoBtn) {
    attDemoPhotoBtn.addEventListener("click", async () => {
      try {
        const res = await fetch(`${API}/api/demo/classroom-photo`);
        const blob = await res.blob();
        const file = new File([blob], "demo_classroom_60.jpg", { type: "image/jpeg" });
        setCapturedImage(file);
        showToast("Loaded 60-Student Demo Classroom Photo! Click 'Run Face Recognition' to test.", "success");
      } catch (err) {
        showToast("Failed to load demo photo: " + err.message, "error");
      }
    });
  }

  document.getElementById("att-recognize-btn").addEventListener("click", runRecognition);

  document.getElementById("att-finalize-btn").addEventListener("click", async () => {
    if (!currentSessionId) return;
    try {
      const res = await fetch(
        `${API}/api/sessions/${currentSessionId}/finalize?class_name=${encodeURIComponent(currentClassName)}`,
        { method: "POST" }
      );
      const data = await res.json();
      showToast(`Session finalized! ${data.records.length} total student records logged.`, "success");

      const exportLink = document.getElementById("att-export-link");
      exportLink.href = `${API}/api/sessions/${currentSessionId}/export`;
      exportLink.style.display = "inline-flex";
      exportLink.setAttribute("download", `attendance_session_${currentSessionId}.csv`);
      loadStats();
    } catch (err) {
      showToast("Finalization error: " + err.message, "error");
    }
  });
}

function setCapturedImage(file) {
  capturedImageFile = file;
  focusedRecordId = null;
  const recognizeBtn = document.getElementById("att-recognize-btn");
  recognizeBtn.disabled = false;

  const canvas = document.getElementById("att-canvas");
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    document.getElementById("att-inspector").classList.remove("hidden");
  };
  img.src = URL.createObjectURL(file);
}

async function runRecognition() {
  if (!capturedImageFile || !currentSessionId) return;

  focusedRecordId = null;
  const recognizeBtn = document.getElementById("att-recognize-btn");
  recognizeBtn.disabled = true;
  recognizeBtn.textContent = "Analyzing Classroom Faces...";

  const threshold = document.getElementById("att-threshold").value;
  const form = new FormData();
  form.append("image", capturedImageFile);

  try {
    const res = await fetch(
      `${API}/api/sessions/${currentSessionId}/recognize?threshold=${threshold}`,
      { method: "POST", body: form }
    );
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Recognition failed");

    lastRecognitionResults = data.results;
    lastRecognitionMeta = { width: data.image_width, height: data.image_height };
    
    updatePhotoCountStats(data.results);
    drawBoundingBoxes(data.results);
    renderResultsCards(data.results);
    
    document.getElementById("att-finalize-row").classList.remove("hidden");
    showToast(`Detected ${data.faces_detected} face(s) and matched ${data.students_matched} student(s)!`, "success");
  } catch (err) {
    showToast("Recognition error: " + err.message, "error");
  } finally {
    recognizeBtn.disabled = false;
    recognizeBtn.textContent = "Run Face Recognition";
  }
}

function updatePhotoCountStats(results = lastRecognitionResults) {
  const summaryBox = document.getElementById("att-photo-summary");
  if (!results || results.length === 0) {
    summaryBox.classList.add("hidden");
    return;
  }

  summaryBox.classList.remove("hidden");

  const totalFaces = results.length;
  const presentCount = results.filter((r) => r.status === "present").length;
  const unknownCount = results.filter((r) => r.status === "unknown").length;

  let classStudentsCount = 0;
  if (currentClassName && allStudentsCache.length > 0) {
    const matchedClass = allStudentsCache.filter(
      (s) => s.class_name && s.class_name.toLowerCase() === currentClassName.toLowerCase()
    );
    classStudentsCount = matchedClass.length > 0 ? matchedClass.length : allStudentsCache.length;
  } else {
    classStudentsCount = allStudentsCache.length;
  }

  const absentCount = Math.max(0, classStudentsCount - presentCount);

  document.getElementById("photo-stat-faces").textContent = totalFaces;
  document.getElementById("photo-stat-present").textContent = presentCount;
  document.getElementById("photo-stat-unknown").textContent = unknownCount;
  document.getElementById("photo-stat-absent").textContent = absentCount;

  document.getElementById("faces-counter-badge").textContent = `${totalFaces} Faces (${presentCount} Matched, ${unknownCount} Unknown)`;
}

function drawBoundingBoxes(results, highlightedRecordId = null, targetFocusId = focusedRecordId) {
  const canvas = document.getElementById("att-canvas");
  const ctx = canvas.getContext("2d");
  const showAllBtn = document.getElementById("btn-show-all-faces");

  if (showAllBtn) {
    if (targetFocusId) {
      showAllBtn.classList.remove("hidden");
    } else {
      showAllBtn.classList.add("hidden");
    }
  }

  // Filter items to draw: if focusedRecordId is set, isolate ONLY that student's face box
  const itemsToDraw = targetFocusId
    ? results.filter((r) => r.record_id === targetFocusId)
    : results;

  // Re-draw original image with matching canvas dimensions
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const scaleX = (lastRecognitionMeta && lastRecognitionMeta.width) ? (img.width / lastRecognitionMeta.width) : 1;
    const scaleY = (lastRecognitionMeta && lastRecognitionMeta.height) ? (img.height / lastRecognitionMeta.height) : 1;

    itemsToDraw.forEach((r) => {
      const x1 = r.bbox[0] * scaleX;
      const y1 = r.bbox[1] * scaleY;
      const x2 = r.bbox[2] * scaleX;
      const y2 = r.bbox[3] * scaleY;
      const isFocused = targetFocusId && r.record_id === targetFocusId;
      const isHighlighted = isFocused || r.record_id === highlightedRecordId;

      const strokeColor = r.status === "present" ? "#10b981" : "#f59e0b";
      ctx.strokeStyle = isHighlighted ? "#6366f1" : strokeColor;
      ctx.lineWidth = isHighlighted ? Math.max(4, canvas.width / 200) : Math.max(2.5, canvas.width / 400);

      // Light semi-transparent overlay box for highlighted / focused face
      if (isHighlighted) {
        ctx.fillStyle = "rgba(99, 102, 241, 0.25)";
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      }

      // Bounding box rectangle
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label background & text
      const label = r.name ? `${r.name} (${(r.confidence * 100).toFixed(0)}%)` : "Unknown Face";
      const fontSize = Math.max(14, Math.round(canvas.width / 45));
      ctx.font = `700 ${fontSize}px Plus Jakarta Sans, sans-serif`;
      const textWidth = ctx.measureText(label).width;

      const padX = 10;
      const padY = 6;
      const tagHeight = fontSize + padY * 2;
      const tagWidth = textWidth + padX * 2;
      const tagY = Math.max(0, y1 - tagHeight - 4);

      ctx.fillStyle = isHighlighted ? "#6366f1" : strokeColor;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(x1, tagY, tagWidth, tagHeight, 4);
      } else {
        ctx.rect(x1, tagY, tagWidth, tagHeight);
      }
      ctx.fill();

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + padX, tagY + fontSize + 1);
    });
  };
  img.src = URL.createObjectURL(capturedImageFile);
}

function studentOptionsHtml(selectedId) {
  let html = `<option value="">-- Mark as Unknown / Not Student --</option>`;
  allStudentsCache.forEach((s) => {
    html += `<option value="${s.student_id}" ${s.student_id === selectedId ? "selected" : ""}>${s.name} (${s.student_id})</option>`;
  });
  return html;
}

function renderResultsCards(results) {
  const container = document.getElementById("att-results");
  container.innerHTML = "";

  results.forEach((r) => {
    const card = document.createElement("div");
    const isFocused = focusedRecordId === r.record_id;
    card.className = `result-card-item ${r.status} ${isFocused ? "focused" : ""}`;
    card.dataset.recordId = r.record_id;

    const confPercent = Math.min(100, Math.max(0, Math.round(r.confidence * 100)));
    const meterClass = confPercent >= 70 ? "high" : confPercent >= 40 ? "medium" : "low";

    card.innerHTML = `
      <div class="result-header">
        <span class="status-badge ${r.status}">${r.status}</span>
        <span class="conf-tag">Similarity: ${confPercent}%</span>
      </div>
      <div class="conf-meter">
        <div class="conf-meter-bar">
          <div class="conf-meter-fill ${meterClass}" style="width: ${confPercent}%"></div>
        </div>
      </div>
      <div class="student-select-row">
        <select class="student-select">${studentOptionsHtml(r.student_id)}</select>
      </div>
      <div class="action-row">
        <button class="btn-link-danger remove-btn">Remove False Detection</button>
      </div>
    `;

    // Click on result card toggles Focus Isolation Mode
    card.addEventListener("click", (e) => {
      if (e.target.tagName === "SELECT" || e.target.tagName === "BUTTON" || e.target.classList.contains("btn-link-danger")) {
        return;
      }

      if (focusedRecordId === r.record_id) {
        focusedRecordId = null;
      } else {
        focusedRecordId = r.record_id;
      }

      document.querySelectorAll(".result-card-item").forEach((item) => {
        if (parseInt(item.dataset.recordId) === focusedRecordId) {
          item.classList.add("focused");
        } else {
          item.classList.remove("focused");
        }
      });

      drawBoundingBoxes(lastRecognitionResults);
    });

    // Hover sync with canvas
    card.addEventListener("mouseenter", () => {
      card.classList.add("highlighted");
      drawBoundingBoxes(lastRecognitionResults, r.record_id);
    });
    card.addEventListener("mouseleave", () => {
      card.classList.remove("highlighted");
      drawBoundingBoxes(lastRecognitionResults);
    });

    // Student selection dropdown override
    card.querySelector(".student-select").addEventListener("change", async (e) => {
      const studentId = e.target.value || null;
      const studentObj = allStudentsCache.find((s) => s.student_id === studentId);
      const name = studentObj ? studentObj.name : null;
      const status = studentId ? "present" : "unknown";

      await fetch(`${API}/api/sessions/${currentSessionId}/records/${r.record_id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ student_id: studentId, name, status }),
      });

      r.student_id = studentId;
      r.name = name;
      r.status = status;

      card.className = `result-card-item ${status}`;
      const badge = card.querySelector(".status-badge");
      badge.className = `status-badge ${status}`;
      badge.textContent = status;

      updatePhotoCountStats(lastRecognitionResults);
      drawBoundingBoxes(lastRecognitionResults);
    });

    // Remove false detection
    card.querySelector(".remove-btn").addEventListener("click", async () => {
      await fetch(`${API}/api/sessions/${currentSessionId}/records/${r.record_id}`, { method: "DELETE" });
      lastRecognitionResults = lastRecognitionResults.filter((item) => item.record_id !== r.record_id);
      card.remove();
      updatePhotoCountStats(lastRecognitionResults);
      drawBoundingBoxes(lastRecognitionResults);
    });

    container.appendChild(card);
  });
}

// ---------------------------------------------------------------------
// Records Tab & Session Viewer
// ---------------------------------------------------------------------
async function loadSessions() {
  const container = document.getElementById("sessions-list");
  try {
    const res = await fetch(`${API}/api/sessions`);
    const sessions = await res.json();
    container.innerHTML = "";

    if (sessions.length === 0) {
      container.innerHTML = `<p style="color: var(--text-muted);">No attendance sessions logged yet.</p>`;
      return;
    }

    sessions.forEach((s) => {
      const card = document.createElement("div");
      card.className = "session-card";
      card.innerHTML = `
        <div class="session-card-header">
          <span class="session-title">Class: ${s.class_name}</span>
          <span class="session-date">${s.session_date}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
          <span class="pill-badge">Session #${s.id}</span>
          <a href="${API}/api/sessions/${s.id}/export" class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;">Export CSV</a>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    showToast("Failed to load sessions: " + err.message, "error");
  }
}

// ---------------------------------------------------------------------
// Student Database Directory & Search
// ---------------------------------------------------------------------
async function loadStudents() {
  try {
    const res = await fetch(`${API}/api/students`);
    const students = await res.json();
    allStudentsCache = students;
    renderStudentsTable(students);
  } catch (err) {
    showToast("Failed to load student list: " + err.message, "error");
  }
}

function renderStudentsTable(students) {
  const tbody = document.getElementById("students-tbody");
  tbody.innerHTML = "";

  if (students.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No registered students found.</td></tr>`;
    return;
  }

  students.forEach((s) => {
    const tr = document.createElement("tr");
    const formattedDate = new Date(s.created_at).toLocaleDateString();
    tr.innerHTML = `
      <td><strong>${s.student_id}</strong></td>
      <td>${s.name}</td>
      <td><span class="pill-badge">${s.class_name || "General"}</span></td>
      <td>${s.photo_count} Photo(s)</td>
      <td>${formattedDate}</td>
      <td>
        <button data-id="${s.student_id}" class="btn-link-danger del-student-btn">Delete Profile</button>
      </td>
    `;

    tr.querySelector(".del-student-btn").addEventListener("click", async () => {
      if (!confirm(`Delete profile for ${s.name} (${s.student_id})?`)) return;
      try {
        await fetch(`${API}/api/students/${s.student_id}`, { method: "DELETE" });
        showToast(`Deleted ${s.name}`, "success");
        loadStudents();
        loadStats();
      } catch (err) {
        showToast("Delete failed: " + err.message, "error");
      }
    });

    tbody.appendChild(tr);
  });
}

function setupStudentSearch() {
  const searchInput = document.getElementById("student-search-input");
  searchInput.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase().trim();
    const filtered = allStudentsCache.filter(
      (s) => s.name.toLowerCase().includes(query) || s.student_id.toLowerCase().includes(query)
    );
    renderStudentsTable(filtered);
  });
}
