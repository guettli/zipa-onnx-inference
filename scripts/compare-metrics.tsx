#!/usr/bin/env tsx
/**
 * Compare three IPA similarity metrics on the pp project's training data.
 *
 * Three metrics:
 *
 * 1. LEVENSHTEIN  — character-level edit distance, normalized by max length.
 *                   Fast, simple, treats all phoneme errors equally.
 *                   Score = 1 − lev(a,b) / max(|a|, |b|)
 *
 * 2. PER          — Phoneme Error Rate (standard ASR metric).
 *                   Normalizes by reference length, not max length.
 *                   Score = max(0, 1 − lev(ref,hyp) / |ref|)
 *                   Penalizes extra insertions more than Levenshtein does.
 *
 * 3. PANPHON      — Feature-weighted phoneme distance (from pp project).
 *                   Substitution cost = fraction of differing articulatory
 *                   features (voicing, place, manner …).  A/ɑ or d/t cost
 *                   little; a/ʃ costs almost 1.  Most linguistically sound.
 *
 * Why compare them?
 *   • Lev vs PER: differ when lengths mismatch (extra insertions/deletions).
 *     PER is harsher when the hypothesis is longer than the reference.
 *   • Both vs PanPhon: differ when the wrong phoneme is acoustically close.
 *     PanPhon rewards near-misses (ɑ/a, t/d, ɾ/r) that the others ignore.
 *
 * Data source: pre-extracted IPA in pp/static/audio/de-DE/edge-tts-male/*.opus.debug.yaml
 */

import fs from 'fs';
import path from 'path';


// ── IPA normalisation ─────────────────────────────────────────────────────────
// Strip olaph-style delimiters (/…/), stress marks, syllable dots,
// tie-bars, spaces so both strings are raw phoneme sequences.
function normalizeIpa(ipa: string): string {
  return ipa
    .replace(/^\/|\/$/g, '')          // strip / delimiters
    .replace(/[ˈˌ.]/g, '')            // stress markers + syllable boundary
    .replace(/\u032F/g, '')            // combining non-syllabic  ̯
    .replace(/\u0329/g, '')            // combining syllabic       ̩
    .replace(/\s+/g, '')               // spaces between words
    .trim();
}

// ── Metric 1: Levenshtein similarity ──────────────────────────────────────────
function lev(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp: number[] = Array.from({ length: n + 1 }, (_, j) => j);
  for (let i = 1; i <= m; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const tmp = dp[j];
      dp[j] = a[i - 1] === b[j - 1] ? prev : 1 + Math.min(prev, dp[j], dp[j - 1]);
      prev = tmp;
    }
  }
  return dp[n];
}

function levSimilarity(ref: string, hyp: string): number {
  if (ref === hyp) return 1;
  const maxLen = Math.max(ref.length, hyp.length);
  if (maxLen === 0) return 1;
  return 1 - lev(ref, hyp) / maxLen;
}

// ── Metric 2: PER (Phoneme Error Rate) ────────────────────────────────────────
function perSimilarity(ref: string, hyp: string): number {
  if (ref === hyp) return 1;
  if (ref.length === 0) return 0;
  return Math.max(0, 1 - lev(ref, hyp) / ref.length);
}

// ── Metric 3: PanPhon ─────────────────────────────────────────────────────────
// Import directly from the pp project build artefacts.
const PP_ROOT = '/home/guettli/projects/pp';

let panphonAvailable = false;
let calculatePanPhonDistance: (ref: string, hyp: string, lang: string) => { similarity: number } =
  () => ({ similarity: 0 });

try {
  const { calculatePanPhonDistance: fn } = await import(
    `${PP_ROOT}/tests/panphon-distance-node.js`
  );
  calculatePanPhonDistance = fn;
  panphonAvailable = true;
} catch (e) {
  process.stderr.write(`Warning: PanPhon unavailable (${(e as Error).message})\n`);
}

function panphonSimilarity(ref: string, hyp: string): number {
  if (!panphonAvailable) return NaN;
  try {
    return calculatePanPhonDistance(ref, hyp, 'de-DE').similarity;
  } catch {
    return NaN;
  }
}

// ── Load data ─────────────────────────────────────────────────────────────────
function parseSimpleYaml(text: string): Record<string, string> {
  const result: Record<string, string> = {};
  for (const line of text.split('\n')) {
    const m = line.match(/^(\w[\w-]*):\s*(.*)$/);
    if (m) result[m[1]] = m[2].trim();
  }
  return result;
}

interface Entry {
  phrase:   string;
  file:     string;
  expected: string;   // normalised
  detected: string;
  lev:      number;
  per:      number;
  pp:       number;
}

const AUDIO_DIR    = `${PP_ROOT}/static/audio/de-DE/edge-tts-male`;
const PHRASES_YAML = `${PP_ROOT}/phrases-de-DE.yaml`;

// Parse phrases-de-DE.yaml (multi-doc YAML with list)
const rawPhrases = fs.readFileSync(PHRASES_YAML, 'utf8');
// Simple extraction: find all phrase + ipa + en-GB
const phraseBlocks: Array<{ phrase: string; ipa: string; enGB: string }> = [];
const lines = rawPhrases.split('\n');
let curPhrase = '', curIpa = '', curEnGB = '';
for (const line of lines) {
  const phraseM = line.match(/^- phrase: (.+)/);
  const ipaM    = line.match(/^\s+- ipa: (.+)/);
  const engM    = line.match(/^\s+en-GB: (.+)/);
  if (phraseM) {
    if (curPhrase && curEnGB) phraseBlocks.push({ phrase: curPhrase, ipa: curIpa, enGB: curEnGB });
    curPhrase = phraseM[1].trim();
    curIpa = ''; curEnGB = '';
  }
  if (ipaM && !curIpa) curIpa = ipaM[1].trim();
  if (engM) curEnGB = engM[1].trim();
}
if (curPhrase && curEnGB) phraseBlocks.push({ phrase: curPhrase, ipa: curIpa, enGB: curEnGB });

const entries: Entry[] = [];

for (const { phrase, ipa: rawIpa, enGB } of phraseBlocks) {
  const stem = enGB.replace(/ /g, '_').replace(/[,.'!?]/g, '');
  const debugPath = path.join(AUDIO_DIR, `${stem}.opus.debug.yaml`);
  if (!fs.existsSync(debugPath)) continue;

  const debug = parseSimpleYaml(fs.readFileSync(debugPath, 'utf8'));
  const detected  = debug['ipa'] ?? '';
  const expected  = normalizeIpa(rawIpa);
  if (!detected || !expected) continue;

  entries.push({
    phrase,
    file: `${stem}.opus`,
    expected,
    detected,
    lev: levSimilarity(expected, detected),
    per: perSimilarity(expected, detected),
    pp:  panphonSimilarity(expected, detected),
  });
}

// ── Metric comparison overview ────────────────────────────────────────────────
const ppValues = entries.map(e => e.pp).filter(v => !isNaN(v));
const avg = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length;

console.log('\n=== Three similarity metrics compared ===\n');
console.log('Metric     Avg     Min     Notes');
console.log('-'.repeat(70));
console.log(
  `Lev       ${(avg(entries.map(e=>e.lev))*100).toFixed(1).padStart(5)}%  ` +
  `${(Math.min(...entries.map(e=>e.lev))*100).toFixed(1).padStart(5)}%  ` +
  `char edit distance / max(|ref|,|hyp|)`,
);
console.log(
  `PER       ${(avg(entries.map(e=>e.per))*100).toFixed(1).padStart(5)}%  ` +
  `${(Math.min(...entries.map(e=>e.per))*100).toFixed(1).padStart(5)}%  ` +
  `edit distance / |ref|  (standard ASR metric, harsher on long hypotheses)`,
);
if (panphonAvailable) {
  console.log(
    `PanPhon   ${(avg(ppValues)*100).toFixed(1).padStart(5)}%  ` +
    `${(Math.min(...ppValues)*100).toFixed(1).padStart(5)}%  ` +
    `feature-weighted substitution cost (phonetically meaningful)`,
  );
}

// Cases where metrics diverge most (PanPhon >> Lev)
if (panphonAvailable) {
  const divergent = entries
    .filter(e => !isNaN(e.pp) && e.lev < 0.9)
    .sort((a, b) => (b.pp - b.lev) - (a.pp - a.lev))
    .slice(0, 5);

  console.log('\n--- Cases where PanPhon scores higher than Levenshtein (near-miss phonemes) ---');
  console.log('Lev   PER   PP    Phrase                Expected               Detected');
  console.log('-'.repeat(100));
  for (const e of divergent) {
    const trunc = (s: string, n: number) => s.length > n ? s.slice(0, n - 1) + '…' : s;
    console.log(
      `${(e.lev*100).toFixed(0).padStart(3)}%  ` +
      `${(e.per*100).toFixed(0).padStart(3)}%  ` +
      `${(e.pp*100).toFixed(0).padStart(3)}%  ` +
      trunc(e.phrase, 24).padEnd(24) + ' ' +
      trunc(e.expected, 22).padEnd(22) + ' ' +
      e.detected,
    );
  }
}

// ── Worst results (sorted by PanPhon if available, else Lev) ─────────────────
const sorted = [...entries].sort((a, b) =>
  panphonAvailable ? a.pp - b.pp : a.lev - b.lev,
);

console.log(`\n=== Worst results (${panphonAvailable ? 'PanPhon' : 'Levenshtein'} score, lowest first) ===\n`);
const N_WORST = 30;
console.log(
  'Lev   PER' +
  (panphonAvailable ? '   PP ' : '') +
  '  Phrase                     Expected IPA             Detected IPA',
);
console.log('-'.repeat(panphonAvailable ? 120 : 105));

for (const e of sorted.slice(0, N_WORST)) {
  const trunc = (s: string, n: number) => s.length > n ? s.slice(0, n - 1) + '…' : s;
  const ppCol = panphonAvailable ? `${(e.pp*100).toFixed(0).padStart(3)}%  ` : '';
  console.log(
    `${(e.lev*100).toFixed(0).padStart(3)}%  ` +
    `${(e.per*100).toFixed(0).padStart(3)}%  ` +
    ppCol +
    trunc(e.phrase, 28).padEnd(28) +
    trunc(e.expected, 24).padEnd(24) +
    e.detected,
  );
}

console.log(`\nTotal: ${entries.length} phrases analysed`);
