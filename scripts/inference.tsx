#!/usr/bin/env tsx
import fs from 'fs';
import { isModelDownloaded, getModelOnnxPath } from '../src/models.js';
import { loadAudio } from '../src/audio.js';
import { runCtcInference } from '../src/inference.js';

const audioFile = process.argv[2];
if (!audioFile) {
  console.error('Usage: inference.tsx <audio-file>');
  process.exit(1);
}
if (!fs.existsSync(audioFile)) {
  console.error(`File not found: ${audioFile}`);
  process.exit(1);
}

const modelId  = 'zipa-small-crctc-300k';
const precision = 'fp32';

if (!isModelDownloaded(modelId, precision)) {
  console.error(`Model ${modelId} not downloaded. Run: ./run tsx scripts/cli.ts --download ${modelId}`);
  process.exit(1);
}

const modelPath = getModelOnnxPath(modelId, precision);
const audio     = loadAudio(audioFile);
const result    = await runCtcInference(modelPath, audio);

// Normalise beam probs relative to this top-K set so they sum to 100%.
const rawProbs = result.beams.map(b => Math.exp(b.logProb));
const total    = rawProbs.reduce((s, p) => s + p, 0);

for (let i = 0; i < result.beams.length; i++) {
  const pct = total > 0 ? ((rawProbs[i] / total) * 100).toFixed(1) : '0.0';
  console.log(`${pct.padStart(5)}%  ${result.beams[i].text}`);
}
