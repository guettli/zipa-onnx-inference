import type { FramePrediction } from './inference.js';
import { computeFrameEnergies } from './audio.js';

// Threshold below which a probability is considered absent (not printed)
const SHOW_THRESHOLD = 0.01;

/**
 * Render the per-frame ASCII debug table, matching the format:
 *
 *   000..035: empty
 *   036: v:00  ⎵:58  d:40
 *   037: v:00        d:98
 *
 * Columns:
 *   v:XX   – normalised RMS energy of that audio segment (0–99)
 *   ⎵:XX   – probability of the space/silence token ▁ (0–99), or blank slot
 *   sym:XX – top-1 IPA token probability (0–99), or absent
 */
export function printDebugOutput(
  frames: FramePrediction[],
  audio: Float32Array,
  ipaText: string,
): void {
  const n = frames.length;
  const energies = computeFrameEnergies(audio, n);

  // Width of frame-index field
  const idxW = String(n - 1).length;

  // Collect rows as strings first so we can collapse empty ranges
  type Row = { empty: true } | { empty: false; cols: string };
  const rows: Row[] = [];

  for (let i = 0; i < n; i++) {
    const f = frames[i];
    const vol = Math.min(99, Math.floor(energies[i] * 99));

    // ⎵ (space token) column — fixed 6-char slot: "⎵:XX  " or "      "
    // Note: ⎵ (U+23B5) is 1 display char on most terminals
    const spacePct = Math.min(99, Math.floor(f.spaceProbability * 100));
    const spaceShown = f.spaceProbability >= SHOW_THRESHOLD;
    const spaceCol = spaceShown
      ? `\u23B5:${String(spacePct).padStart(2, '0')}`  // ⎵:XX
      : '      ';                                        // 6 spaces

    // top IPA token column
    const topCol = f.topToken && f.topToken.probability >= SHOW_THRESHOLD
      ? `${f.topToken.symbol}:${String(Math.min(99, Math.floor(f.topToken.probability * 100))).padStart(2, '0')}`
      : '';

    // Is this frame "empty"? = blank dominates (energy low AND no real predictions)
    const isEmpty = !spaceShown && !topCol;

    if (isEmpty) {
      rows.push({ empty: true });
    } else {
      const volCol  = `v:${String(vol).padStart(2, '0')}`;
      rows.push({ empty: false, cols: `${volCol}  ${spaceCol}  ${topCol}`.trimEnd() });
    }
  }

  // Print, collapsing consecutive empty frames into a range
  let emptyStart = -1;

  function flushEmpty(upTo: number) {
    if (emptyStart < 0) return;
    const s = String(emptyStart).padStart(idxW, '0');
    const e = String(upTo - 1).padStart(idxW, '0');
    if (emptyStart === upTo - 1) {
      console.log(`${s}: empty`);
    } else {
      console.log(`${s}..${e}: empty`);
    }
    emptyStart = -1;
  }

  for (let i = 0; i < n; i++) {
    const row = rows[i];
    if (row.empty) {
      if (emptyStart < 0) emptyStart = i;
    } else {
      flushEmpty(i);
      const idx = String(i).padStart(idxW, '0');
      console.log(`${idx}: ${row.cols}`);
    }
  }
  flushEmpty(n);

  console.log('');
  console.log(`IPA: ${ipaText}`);
}
