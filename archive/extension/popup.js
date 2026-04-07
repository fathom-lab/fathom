// FATHOM Chrome Extension — Popup Script
// ========================================
// Shows last scored result

function gradeClass(grade) {
  if (!grade) return "";
  if (grade.startsWith("S") || grade.startsWith("A")) return "grade-A";
  if (grade.startsWith("B")) return "grade-B";
  if (grade.startsWith("C")) return "grade-C";
  return "grade-D";
}

function render(score) {
  const el = document.getElementById("content");

  if (!score) {
    el.innerHTML = `
      <div class="empty">
        Select AI-generated text.<br>
        Right-click &gt; "Score with Fathom"
      </div>
    `;
    return;
  }

  const cls = gradeClass(score.grade);
  const ago = timeSince(score.timestamp);

  el.innerHTML = `
    <div class="score-display">
      <div class="grade-badge ${cls}">${score.grade || "?"}</div>
      <div class="score-number">${score.score != null ? score.score + "/100" : "—"}</div>
    </div>
    <div class="interpretation">${score.interpretation || ""}</div>
    <div class="details">
      <div class="detail-row">
        <span class="detail-label">Depth Score</span>
        <span class="detail-value">${score.depth_score != null ? score.depth_score.toFixed(3) : "—"}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Entropy</span>
        <span class="detail-value">${score.entropy != null ? score.entropy.toFixed(2) : "—"}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Scored</span>
        <span class="detail-value">${ago}</span>
      </div>
    </div>
    ${score.text ? `<div class="text-preview">"${score.text}..."</div>` : ""}
  `;
}

function timeSince(ts) {
  if (!ts) return "—";
  const seconds = Math.floor((Date.now() - ts) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return Math.floor(seconds / 60) + "m ago";
  if (seconds < 86400) return Math.floor(seconds / 3600) + "h ago";
  return Math.floor(seconds / 86400) + "d ago";
}

// Load last score from storage
chrome.storage.local.get("lastScore", (data) => {
  render(data.lastScore || null);
});
