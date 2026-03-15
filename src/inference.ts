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

export interface InferenceResult {
  /** CTC-decoded IPA string */
  ipaText: string;
  /** Per-output-frame predictions */
  frames: FramePrediction[];
  /** Number of Fbank input frames (before subsampling) */
  numFbankFrames: number;
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
  return { ipaText, frames, numFbankFrames: numFrames };
}
