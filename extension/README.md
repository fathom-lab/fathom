# Fathom Chrome Extension

Score any AI-generated text for reasoning depth. Right-click. See the grade.

## Install

1. Start the Fathom API locally:
   ```
   cd sae-reasoning-depth
   uvicorn depth_api_v2:app --host 127.0.0.1 --port 8400
   ```

2. Load the extension in Chrome:
   - Open `chrome://extensions/`
   - Enable "Developer mode" (top right)
   - Click "Load unpacked"
   - Select the `extension/` folder

3. The green **F** icon appears in your toolbar.

## Usage

1. Select any AI-generated text on a webpage
2. Right-click > **"Score with Fathom"**
3. A badge appears near the selection showing the grade:
   - **A+ / A** (green) — Deep distributed reasoning
   - **B+ / B** (gold) — Moderate reasoning depth
   - **C** (orange) — Shallow processing
   - **D / F** (red) — Surface-level pattern matching
4. Click the Fathom icon in the toolbar to see the full score details

The badge fades after 5 seconds. Click anywhere to dismiss it early.

## Requirements

- Fathom API running on `localhost:8400`
- Chrome 110+ (Manifest V3)

## How It Works

The extension sends the selected text to `POST /v2/fathom_score` on your local Fathom API. The API runs circuit attribution through SAE transcoders and returns a depth score, grade, and interpretation.

All processing happens on your local machine. No text is sent to any external server.

## Files

```
extension/
  manifest.json    — Chrome Manifest V3 config
  background.js    — Service worker: context menu + API calls
  content.js       — Badge overlay on pages
  popup.html       — Extension popup UI
  popup.js         — Popup logic (shows last score)
  icons/           — Extension icons (16, 48, 128px)
  README.md        — This file
```
