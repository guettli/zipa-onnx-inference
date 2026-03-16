#!/usr/bin/env tsx
/**
 * Run IPA inference on all data/de-DE test cases and report results,
 * sorted worst-first.
 *
 * Score = probability-weighted Levenshtein similarity:
 *   Σ  P(beam)  ×  sim(beam.text, expected)
 *
 * This is the expected similarity under the model's full output distribution,
 * not just the greedy-decode winner.
 */
import fs from 'fs';
import path from 'path';
import { isModelDownloaded, getModelOnnxPath } from '../src/models.js';
import { loadAudio } from '../src/audio.js';
import { runCtcInference, probWeightedSimilarity } from '../src/inference.js';

// ── Levenshtein similarity ─────────────────────────────────────────────────────

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

function levSim(a: string, b: string): number {
  if (a === b) return 1;
  const maxLen = Math.max(a.length, b.length);
  return maxLen === 0 ? 1 : 1 - lev(a, b) / maxLen;
}

// ── YAML reader ────────────────────────────────────────────────────────────────

function readSimpleYaml(filePath: string): Record<string, string> {
  const result: Record<string, string> = {};
  for (const line of fs.readFileSync(filePath, 'utf8').split('\n')) {
    const m = line.match(/^(\w[\w-]*):\s*(.*)$/);
    if (m) result[m[1]] = m[2].trim();
  }
  return result;
}

// ── Main ───────────────────────────────────────────────────────────────────────

const MODEL_ID = 'zipa-small-crctc-300k';
const PRECISION = 'fp32';
const DATA_DIR  = 'data/de-DE';

if (!isModelDownloaded(MODEL_ID, PRECISION)) {
  console.error(`Model ${MODEL_ID} not downloaded.`);
  console.error(`Run: ./run tsx scripts/cli.ts --download ${MODEL_ID}`);
  process.exit(1);
}

const modelPath = getModelOnnxPath(MODEL_ID, PRECISION);

const yamlFiles = fs.readdirSync(DATA_DIR, { recursive: true, withFileTypes: true })
  .filter(e => e.isFile() && e.name.endsWith('.flac.yaml'))
  .map(e => path.join(e.parentPath ?? e.path, e.name));

if (yamlFiles.length === 0) {
  console.error(`No test data found in ${DATA_DIR}`);
  process.exit(1);
}

interface Result {
  file:     string;
  phrase:   string;
  expected: string;
  greedy:   string;
  bestBeam: string;
  score:    number;   // probability-weighted similarity
  greedyScore: number;
}

const results: Result[] = [];
let idx = 0;

for (const yamlPath of yamlFiles) {
  idx++;
  const flacPath = yamlPath.replace(/\.yaml$/, '');
  if (!fs.existsSync(flacPath)) continue;

  const meta     = readSimpleYaml(yamlPath);
  const expected = meta['recognized_ipa'] ?? '';
  const phrase   = meta['phrase'] ?? path.basename(flacPath);

  process.stderr.write(`[${String(idx).padStart(2)}/${yamlFiles.length}] ${phrase} ... `);

  const audio  = loadAudio(flacPath);
  const result = await runCtcInference(modelPath, audio);

  const score       = probWeightedSimilarity(result.beams, expected, levSim);
  const greedyScore = levSim(result.ipaText, expected);
  const bestBeam    = result.beams[0]?.text ?? result.ipaText;

  process.stderr.write(`pw=${(score * 100).toFixed(0)}%  greedy=${(greedyScore * 100).toFixed(0)}%\n`);

  results.push({ file: path.relative(DATA_DIR, flacPath), phrase, expected, greedy: result.ipaText, bestBeam, score, greedyScore });
}

// Sort worst (lowest prob-weighted score) first.
results.sort((a, b) => a.score - b.score);

const colW = { file: 34, phrase: 20, expected: 20, greedy: 20 };
const trunc = (s: string, n: number) => s.length > n ? s.slice(0, n - 1) + '…' : s;

console.log('\n');
console.log(
  'PW     GRD   ' +
  'FILE'.padEnd(colW.file) +
  'PHRASE'.padEnd(colW.phrase) +
  'EXPECTED'.padEnd(colW.expected) +
  'GREEDY'.padEnd(colW.greedy) +
  'BEST BEAM',
);
console.log('-'.repeat(14 + colW.file + colW.phrase + colW.expected + colW.greedy + 22));

for (const r of results) {
  const pw  = `${(r.score * 100).toFixed(0)}%`.padStart(4);
  const grd = `${(r.greedyScore * 100).toFixed(0)}%`.padStart(4);
  console.log(
    `${pw}   ${grd}   ` +
    trunc(r.file, colW.file).padEnd(colW.file) +
    trunc(r.phrase, colW.phrase).padEnd(colW.phrase) +
    trunc(r.expected, colW.expected).padEnd(colW.expected) +
    trunc(r.greedy, colW.greedy).padEnd(colW.greedy) +
    r.bestBeam,
  );
}

const avgPw  = results.reduce((s, r) => s + r.score, 0) / results.length;
const avgGrd = results.reduce((s, r) => s + r.greedyScore, 0) / results.length;
const improved = results.filter(r => r.score > r.greedyScore).length;

console.log('');
console.log(`Prob-weighted avg: ${(avgPw * 100).toFixed(1)}%   Greedy avg: ${(avgGrd * 100).toFixed(1)}%   (${results.length} files, ${improved} improved by beam search)`);
