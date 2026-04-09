/**
 * Record fathom_demo_video.html → high-quality MP4 for Twitter
 * Captures frames via Puppeteer headless Chrome, assembles with ffmpeg.
 *
 * Usage: node record_demo.js
 * Output: fathom_demo.mp4 (tweet-ready)
 */
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const URL = 'http://localhost:8777/fathom_demo_video.html';
const FRAME_DIR = path.join(__dirname, '_frames');
const OUTPUT_MP4 = path.join(__dirname, 'fathom_demo.mp4');
const OUTPUT_GIF = path.join(__dirname, 'fathom_demo.gif');
const WIDTH = 900;
const HEIGHT = 520;
const FPS = 10;           // capture rate — smooth enough, keeps file small
const DURATION_SEC = 44;  // total recording length
const TOTAL_FRAMES = FPS * DURATION_SEC;

async function main() {
  // Clean
  if (fs.existsSync(FRAME_DIR)) fs.rmSync(FRAME_DIR, { recursive: true });
  fs.mkdirSync(FRAME_DIR);

  console.log('[1/4] launching headless chrome...');
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 2 });

  console.log('[2/4] navigating to demo page...');
  await page.goto(URL, { waitUntil: 'networkidle0', timeout: 15000 });

  console.log(`[3/4] capturing ${TOTAL_FRAMES} frames at ${FPS}fps (~${DURATION_SEC}s)...`);
  const interval = 1000 / FPS;
  const t0 = Date.now();
  for (let i = 0; i < TOTAL_FRAMES; i++) {
    const framePath = path.join(FRAME_DIR, `frame_${String(i).padStart(4, '0')}.png`);
    await page.screenshot({ path: framePath, type: 'png' });
    // Wait remaining time in interval
    const elapsed = Date.now() - t0;
    const target = (i + 1) * interval;
    const sleepMs = Math.max(0, target - elapsed);
    if (sleepMs > 0) await new Promise(r => setTimeout(r, sleepMs));
    if (i % (FPS * 5) === 0) {
      const pct = ((i / TOTAL_FRAMES) * 100).toFixed(0);
      console.log(`  ${pct}% (frame ${i}/${TOTAL_FRAMES})`);
    }
  }
  const captureTime = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`  100% — captured in ${captureTime}s`);

  await browser.close();

  console.log('[4/4] encoding with ffmpeg...');

  // MP4 — H.264, high quality, Twitter-compatible
  execSync([
    'ffmpeg -y',
    `-framerate ${FPS}`,
    `-i "${FRAME_DIR}/frame_%04d.png"`,
    '-c:v libx264',
    '-preset slow',
    '-crf 18',
    '-pix_fmt yuv420p',
    '-movflags +faststart',     // Twitter needs this for streaming
    `-vf "scale=${WIDTH}:-2"`,  // ensure even dimensions
    `"${OUTPUT_MP4}"`,
  ].join(' '), { stdio: 'inherit' });

  const mp4Size = (fs.statSync(OUTPUT_MP4).size / 1024 / 1024).toFixed(1);
  console.log(`\n  MP4: ${OUTPUT_MP4} (${mp4Size} MB)`);

  // Also make a GIF (for embedding/previews)
  try {
    const palette = path.join(FRAME_DIR, 'palette.png');
    execSync(
      `ffmpeg -y -framerate ${FPS} -i "${FRAME_DIR}/frame_%04d.png" -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,palettegen=max_colors=96" "${palette}"`,
      { stdio: 'pipe' }
    );
    execSync(
      `ffmpeg -y -framerate ${FPS} -i "${FRAME_DIR}/frame_%04d.png" -i "${palette}" -lavfi "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" "${OUTPUT_GIF}"`,
      { stdio: 'pipe' }
    );
    const gifSize = (fs.statSync(OUTPUT_GIF).size / 1024 / 1024).toFixed(1);
    console.log(`  GIF: ${OUTPUT_GIF} (${gifSize} MB)`);
  } catch (e) {
    console.log('  GIF encoding failed (ffmpeg issue), MP4 is fine for Twitter.');
  }

  // Cleanup frames
  fs.rmSync(FRAME_DIR, { recursive: true });
  console.log('\n  frames cleaned up. done.');
  console.log('\n  upload the MP4 to Twitter — it handles MP4 much better than GIF.');
}

main().catch(console.error);
