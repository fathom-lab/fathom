// FATHOM Chrome Extension — Background Service Worker
// ====================================================
// Registers context menu and handles scoring via local API

const FATHOM_API = "http://localhost:8400";

// Register context menu on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "fathom-score",
    title: "Score with Fathom",
    contexts: ["selection"],
  });
});

// Handle context menu click
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "fathom-score") return;

  const selectedText = info.selectionText;
  if (!selectedText || selectedText.trim().length < 5) {
    chrome.tabs.sendMessage(tab.id, {
      type: "fathom-error",
      message: "Select more text to score.",
    });
    return;
  }

  // Show loading state
  chrome.tabs.sendMessage(tab.id, {
    type: "fathom-loading",
  });

  try {
    const response = await fetch(`${FATHOM_API}/v2/fathom_score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: selectedText.trim().slice(0, 4096) }),
    });

    if (!response.ok) {
      throw new Error(`API returned ${response.status}`);
    }

    const data = await response.json();
    const result = data.data || data;

    const scoreResult = {
      score: result.fathom_score,
      grade: result.grade,
      interpretation: result.interpretation,
      depth_score: result.depth_result?.depth_score,
      entropy: result.depth_result?.layer_entropy,
    };

    // Save last score for popup
    chrome.storage.local.set({
      lastScore: {
        ...scoreResult,
        text: selectedText.slice(0, 200),
        timestamp: Date.now(),
      },
    });

    // Send to content script for badge overlay
    chrome.tabs.sendMessage(tab.id, {
      type: "fathom-result",
      ...scoreResult,
    });
  } catch (err) {
    chrome.tabs.sendMessage(tab.id, {
      type: "fathom-error",
      message: "Fathom API unreachable. Is it running on localhost:8400?",
    });
  }
});
