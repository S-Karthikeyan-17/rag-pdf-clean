// ============================
// script.js
// ============================

let docId = null;

const uploadBtn = document.getElementById("uploadBtn");
const askBtn = document.getElementById("askBtn");
const uploadStatus = document.getElementById("uploadStatus");
const questionInput = document.getElementById("question");
const chatSection = document.getElementById("chat-section");
const chatWindow = document.getElementById("chat-window");
const contextsBox = document.getElementById("contexts");

/**
 * Append a chat message to the chat window
 */
function appendMessage(text, sender) {
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");
  msgDiv.textContent = text;
  chatWindow.appendChild(msgDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

/**
 * Handle PDF upload
 */
uploadBtn.addEventListener("click", async () => {
  const fileInput = document.getElementById("pdfFile");
  if (!fileInput.files.length) {
    alert("Please select a PDF file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  uploadStatus.textContent = "📤 Uploading...";
  uploadBtn.disabled = true;

  try {
    const res = await fetch("/documents", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());

    const data = await res.json();
    docId = data.doc_id;

    uploadStatus.textContent = `✅ Uploaded! Pages: ${data.pages}, Chunks: ${data.chunks}`;
    chatSection.classList.remove("hidden");

    appendMessage("Hi! Ask me anything about your PDF. 👋", "bot");
  } catch (err) {
    uploadStatus.textContent = "❌ Upload failed.";
    console.error("Upload error:", err);
  } finally {
    uploadBtn.disabled = false;
  }
});

/**
 * Ask a question
 */
askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});

async function askQuestion() {
  if (!docId) {
    alert("Please upload a PDF first.");
    return;
  }

  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage(question, "user");
  questionInput.value = "";

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: docId, question })
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    appendMessage(data.answer, "bot");

    // Show retrieved sources
    if (data.contexts && data.contexts.length > 0) {
      contextsBox.classList.remove("hidden");
      let ctxHTML = "<strong>📚 Sources:</strong>";
      data.contexts.forEach(c => {
        ctxHTML += `
          <p>
            <strong>Page ${c.page}</strong> (score: ${(c.score * 100).toFixed(1)}%):<br>
            ${escapeHTML(c.text)}
          </p>`;
      });
      contextsBox.innerHTML = ctxHTML;
    }
  } catch (err) {
    appendMessage("❌ Error fetching answer.", "bot");
    console.error("Query error:", err);
  }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHTML(str) {
  return str.replace(/[&<>'"]/g, tag => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[tag]
  ));
}
