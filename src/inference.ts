import * as ort from 'onnxruntime-node';
import { extractFbank } from './fbank.js';
import { BLANK_ID, SPACE_ID, ID_TO_TOKEN } from './tokens.js';

export interface FramePrediction {
  /** Softmax probability of the space/silence token (▁), 0-1 */
  spaceProbability: number;
  /** Top non-blank, non-space token and its probability */
  topToken: { symbol: string; probability: number } | null;
  /** Raw probabilities for all tokens (length = vocab_size) */
  probs: Float32Array;
}

export interface Beam {
  /** CTC-decoded IPA text for this hypothesis */
  text: string;
  /** Natural log of the hypothesis probability */
  logProb: number;
}

export interface InferenceResult {
  /** CTC greedy-decoded IPA string */
  ipaText: string;
  /** Per-output-frame predictions */
  frames: FramePrediction[];
  /** Number of Fbank input frames (before subsampling) */
  numFbankFrames: number;
  /** Top-K beam search hypotheses, sorted by logProb descending */
  beams: Beam[];
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Numerically stable log(exp(a) + exp(b)). */
function logSumExp(a: number, b: number): number {
  if (!isFinite(a)) return b;
  if (!isFinite(b)) return a;
  const max = Math.max(a, b);
  return max + Math.log1p(Math.exp(Math.min(a, b) - max));
}

/**
 * CTC prefix beam search.
 *
 * Returns up to `beamWidth` hypotheses sorted by log-probability descending.
 * Works in log space throughout to avoid underflow on longer sequences.
 *
 * Algorithm: Graves et al. 2006, "Connectionist Temporal Classification"
 * Each beam tracks:
 *   logPb  – log P(prefix ends in blank at current timestep)
 *   logPnb – log P(prefix ends in non-blank at current timestep)
 *   lastId – token ID of the last emitted (non-blank) symbol
 */
export function ctcBeamSearch(
  frames: FramePrediction[],
  beamWidth = 10,
): Beam[] {
  // Pre-compute which token IDs produce real IPA output.
  const emittable: Array<{ id: number; out: string }> = [];
  for (const [idStr, sym] of Object.entries(ID_TO_TOKEN)) {
    const id = Number(idStr);
    if (id === BLANK_ID) continue;
    if (sym.startsWith('<')) continue; // <sos/eos>, <unk>
    emittable.push({ id, out: sym === '▁' ? ' ' : sym });
  }

  interface BeamState { logPb: number; logPnb: number; lastId: number }
  const LOG_ZERO = -Infinity;

  // beamMap key = output prefix text (with spaces for ▁)
  let beamMap = new Map<string, BeamState>([
    ['', { logPb: 0, logPnb: LOG_ZERO, lastId: -1 }],
  ]);

  for (const frame of frames) {
    const logP = new Float64Array(frame.probs.length);
    for (let i = 0; i < frame.probs.length; i++) {
      logP[i] = frame.probs[i] > 0 ? Math.log(frame.probs[i]) : LOG_ZERO;
    }

    const next = new Map<string, BeamState>();

    const add = (key: string, lastId: number, dPb: number, dPnb: number) => {
      const ex = next.get(key);
      if (ex === undefined) {
        next.set(key, { logPb: dPb, logPnb: dPnb, lastId });
      } else {
        ex.logPb = logSumExp(ex.logPb, dPb);
        ex.logPnb = logSumExp(ex.logPnb, dPnb);
      }
    };

    const logPBlank = logP[BLANK_ID];

    for (const [prefix, { logPb, logPnb, lastId }] of beamMap) {
      const logPTotal = logSumExp(logPb, logPnb);

      // Extend with blank: prefix unchanged, pb accumulates.
      add(prefix, lastId, logPTotal + logPBlank, LOG_ZERO);

      // Extend with each emittable token.
      for (const { id, out } of emittable) {
        const logPToken = logP[id];
        if (id === lastId) {
          // Same token as last emitted:
          //   from pb (prev ended in blank)  → adds new char → prefix + out
          add(prefix + out, id, LOG_ZERO, logPb + logPToken);
          //   from pnb (prev ended non-blank) → continues run → stays at prefix
          add(prefix, lastId, LOG_ZERO, logPnb + logPToken);
        } else {
          // Different token → always appends to prefix.
          add(prefix + out, id, LOG_ZERO, logPTotal + logPToken);
        }
      }
    }

    // Prune to top beamWidth by total log probability.
    beamMap = new Map(
      [...next.entries()]
        .sort((a, b) =>
          logSumExp(b[1].logPb, b[1].logPnb) -
          logSumExp(a[1].logPb, a[1].logPnb),
        )
        .slice(0, beamWidth),
    );
  }

  return [...beamMap.entries()]
    .map(([text, { logPb, logPnb }]) => ({
      text: text.trim(),
      logProb: logSumExp(logPb, logPnb),
    }))
    .sort((a, b) => b.logProb - a.logProb);
}

/**
 * Compute the probability-weighted similarity between a set of beam hypotheses
 * and a reference string.
 *
 * score = Σ  P(beam) × sim(beam.text, reference)
 *
 * This is the expected similarity under the model's output distribution,
 * which is more informative than greedy-decode similarity alone:
 * it rewards cases where the correct answer has high probability mass
 * even if it is not the single most likely hypothesis.
 */
export function probWeightedSimilarity(
  beams: Beam[],
  reference: string,
  simFn: (a: string, b: string) => number,
): number {
  if (beams.length === 0) return 0;
  // Normalise: beam probs may not sum to 1 after pruning.
  const probs = beams.map(b => Math.exp(b.logProb));
  const total = probs.reduce((s, p) => s + p, 0);
  if (total === 0) return 0;
  return beams.reduce((sum, b, i) => sum + (probs[i] / total) * simFn(b.text, reference), 0);
}

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const exp = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { exp[i] = Math.exp(logits[i] - max); sum += exp[i]; }
  for (let i = 0; i < logits.length; i++) exp[i] /= sum;
  return exp;
}

function ctcGreedyDecode(frames: FramePrediction[]): string {
  const tokens: string[] = [];
  let prevId = -1;
  for (const frame of frames) {
    // argmax over all probs
    let maxProb = -Infinity;
    let maxId = 0;
    for (let i = 0; i < frame.probs.length; i++) {
      if (frame.probs[i] > maxProb) { maxProb = frame.probs[i]; maxId = i; }
    }
    if (maxId !== BLANK_ID && maxId !== prevId) {
      const sym = ID_TO_TOKEN[maxId] ?? '';
      if (sym && sym !== '<blk>' && sym !== '<sos/eos>' && sym !== '<unk>') {
        tokens.push(sym === '▁' ? ' ' : sym);
      }
    }
    prevId = maxId;
  }
  return tokens.join('').trim();
}

export async function runCtcInference(
  modelPath: string,
  audio: Float32Array,
): Promise<InferenceResult> {
  // Feature extraction
  const { features, numFrames } = extractFbank(audio);

  // ONNX session
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['cpu'],
  });

  const inputTensor = new ort.Tensor('float32', features, [1, numFrames, 80]);
  const lensTensor  = new ort.Tensor('int64', new BigInt64Array([BigInt(numFrames)]), [1]);

  const outputs = await session.run({ x: inputTensor, x_lens: lensTensor });
  const outputName = session.outputNames[0];
  const rawOutput  = outputs[outputName];

  // rawOutput shape: [1, T_out, vocab_size]
  const [, numCtcFrames, vocabSize] = rawOutput.dims as number[];
  const rawData = rawOutput.data as Float32Array;

  // Build per-frame predictions
  const frames: FramePrediction[] = [];
  for (let f = 0; f < numCtcFrames; f++) {
    const logits = rawData.slice(f * vocabSize, (f + 1) * vocabSize);
    const probs  = softmax(logits);

    const spaceProbability = probs[SPACE_ID] ?? 0;

    // Top non-blank, non-space token
    let topProb = 0;
    let topId   = -1;
    for (let i = 0; i < probs.length; i++) {
      if (i === BLANK_ID || i === SPACE_ID) continue;
      if (probs[i] > topProb) { topProb = probs[i]; topId = i; }
    }
    const topToken = topId >= 0 && topProb > 0.001
      ? { symbol: ID_TO_TOKEN[topId] ?? `?${topId}`, probability: topProb }
      : null;

    frames.push({ spaceProbability, topToken, probs });
  }

  const ipaText = ctcGreedyDecode(frames);
  const beams = ctcBeamSearch(frames);
  return { ipaText, frames, numFbankFrames: numFrames, beams };
}
