// FATHOM Chrome Extension — Content Script
// ==========================================
// Shows badge overlay near selected text after scoring

(function () {
  let currentBadge = null;

  function removeBadge() {
    if (currentBadge) {
      currentBadge.remove();
      currentBadge = null;
    }
  }

  function gradeColor(grade) {
    if (!grade) return "#666";
    if (grade.startsWith("S") || grade.startsWith("A")) return "#00ff00";
    if (grade.startsWith("B")) return "#ffd700";
    if (grade.startsWith("C")) return "#ff9800";
    if (grade.startsWith("D")) return "#ff3d00";
    return "#ff3d00";
  }

  function createBadge(content, color) {
    removeBadge();

    const sel = window.getSelection();
    if (!sel.rangeCount) return;

    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();

    const badge = document.createElement("div");
    badge.id = "fathom-badge";
    badge.style.cssText = `
      position: fixed;
      top: ${Math.max(8, rect.top - 44)}px;
      left: ${Math.min(rect.left, window.innerWidth - 200)}px;
      background: #0a0a0a;
      border: 1px solid ${color};
      color: ${color};
      font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
      font-size: 12px;
      padding: 6px 12px;
      border-radius: 4px;
      z-index: 2147483647;
      box-shadow: 0 2px 12px rgba(0,0,0,0.5);
      pointer-events: none;
      transition: opacity 0.3s;
      white-space: nowrap;
    `;
    badge.innerHTML = content;

    document.body.appendChild(badge);
    currentBadge = badge;

    // Fade out after 5 seconds
    setTimeout(() => {
      if (badge && badge.parentNode) {
        badge.style.opacity = "0";
        setTimeout(() => badge.remove(), 300);
        if (currentBadge === badge) currentBadge = null;
      }
    }, 5000);
  }

  // Listen for messages from background
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "fathom-loading") {
      createBadge("FATHOM scoring...", "#666");
    } else if (msg.type === "fathom-result") {
      const color = gradeColor(msg.grade);
      const scoreText = msg.score != null ? msg.score : "?";
      const gradeText = msg.grade || "?";
      createBadge(
        `<strong>FATHOM</strong> ${gradeText} &middot; ${scoreText}/100`,
        color
      );
    } else if (msg.type === "fathom-error") {
      createBadge(msg.message || "Error", "#ff3d00");
    }
  });

  // Remove badge on click
  document.addEventListener("click", removeBadge);
})();
