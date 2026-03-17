/**
 * Web Worker: handles model download, ONNX inference, beam search.
 * Runs off the main thread so the UI stays responsive.
 *
 * Option A: ort.env.wasm.numThreads = 1  → no SharedArrayBuffer required.
 */
import * as ort from 'onnxruntime-web';
import { extractFbank } from '../../src/fbank.js';
import { BLANK_ID, ID_TO_TOKEN } from '../../src/tokens.js';
import { getModelFromCache, saveModelToCache, getPartialDownload, savePartialDownload, clearPartialDownload } from './model-cache.js';
import type { ModelInfo, WorkerInMsg, WorkerOutMsg, FrameOut, BeamOut } from './types.js';
import { MODELS, modelUrl } from './types.js';

// ── Option A: single-threaded, no SharedArrayBuffer needed ────────────────────
ort.env.wasm.numThreads = 1;
// Load WASM files from CDN so we don't have to bundle them.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/';

// ── State ─────────────────────────────────────────────────────────────────────

let session: ort.InferenceSession | null = null;
let loadedModelId: string | null = null;
let cancelled = false;

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Download with resumable partial saves every 5 MB. */
async function downloadWithProgress(url: string, onProgress: (pct: number, mb: number, total: number) => void): Promise<ArrayBuffer> {
  const partial = await getPartialDownload(url);
  const headers: Record<string, string> = {};
  let received = partial ? partial.data.byteLength : 0;
  if (partial) headers['Range'] = `bytes=${received}-`;

  let resp: Response;
  try {
    resp = await fetch(url, { headers });
  } catch {
    throw new Error(`Network error — check your internet connection and try again.`);
  }
  if (resp.status === 404) throw new Error(`Model file not found (404). The model may have moved on Hugging Face.`);
  if (resp.status === 429) throw new Error(`Rate limited by Hugging Face (429). Please wait a moment and try again.`);
  if (!resp.ok && resp.status !== 206) throw new Error(`Download failed (HTTP ${resp.status}). Try reloading the page.`);

  const total = partial
    ? partial.total
    : parseInt(resp.headers.get('content-length') ?? '0', 10);

  const reader = resp.body!.getReader();
  const chunks: Uint8Array[] = partial ? [new Uint8Array(partial.data)] : [];
  let lastSave = received;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress(total > 0 ? received / total : 0, received / 1e6, total / 1e6);

    // Save partial every 5 MB
    if (received - lastSave > 5 * 1e6) {
      const merged = mergeChunks(chunks, received);
      await savePartialDownload(url, { data: merged, total });
      lastSave = received;
    }
  }

  const full = mergeChunks(chunks, received);
  await clearPartialDownload(url);
  return full;
}

function mergeChunks(chunks: Uint8Array[], totalBytes: number): ArrayBuffer {
  const out = new Uint8Array(totalBytes);
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.byteLength; }
  return out.buffer;
}

// ── Softmax ───────────────────────────────────────────────────────────────────

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const exp = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) { exp[i] = Math.exp(logits[i] - max); sum += exp[i]; }
  for (let i = 0; i < logits.length; i++) exp[i] /= sum;
  return exp;
}

// ── CTC beam search (same algorithm as Node.js src/inference.ts) ──────────────

function logSumExp(a: number, b: number): number {
  if (!isFinite(a)) return b;
  if (!isFinite(b)) return a;
  const max = Math.max(a, b);
  return max + Math.log1p(Math.exp(Math.min(a, b) - max));
}

interface BeamState { logPb: number; logPnb: number; lastId: number }

const EMITTABLE: Array<{ id: number; out: string }> = (() => {
  const result: Array<{ id: number; out: string }> = [];
  for (const [idStr, sym] of Object.entries(ID_TO_TOKEN)) {
    const id = Number(idStr);
    if (id === BLANK_ID) continue;
    if (sym.startsWith('<')) continue;
    result.push({ id, out: sym === '▁' ? ' ' : sym });
  }
  return result;
})();

function ctcBeamSearch(frameProbs: Float32Array[], beamWidth = 10): BeamOut[] {
  const LOG_ZERO = -Infinity;
  let beamMap = new Map<string, BeamState>([
    ['', { logPb: 0, logPnb: LOG_ZERO, lastId: -1 }],
  ]);

  for (const probs of frameProbs) {
    const logP = new Float64Array(probs.length);
    for (let i = 0; i < probs.length; i++) logP[i] = probs[i] > 0 ? Math.log(probs[i]) : LOG_ZERO;

    const next = new Map<string, BeamState>();
    const add = (key: string, lastId: number, dPb: number, dPnb: number) => {
      const ex = next.get(key);
      if (!ex) { next.set(key, { logPb: dPb, logPnb: dPnb, lastId }); }
      else { ex.logPb = logSumExp(ex.logPb, dPb); ex.logPnb = logSumExp(ex.logPnb, dPnb); }
    };

    for (const [prefix, { logPb, logPnb, lastId }] of beamMap) {
      const logPTotal = logSumExp(logPb, logPnb);
      add(prefix, lastId, logPTotal + logP[BLANK_ID], LOG_ZERO);
      for (const { id, out } of EMITTABLE) {
        const logPToken = logP[id];
        if (id === lastId) {
          add(prefix + out, id, LOG_ZERO, logPb + logPToken);
          add(prefix, lastId, LOG_ZERO, logPnb + logPToken);
        } else {
          add(prefix + out, id, LOG_ZERO, logPTotal + logPToken);
        }
      }
    }

    beamMap = new Map(
      [...next.entries()]
        .sort((a, b) => logSumExp(b[1].logPb, b[1].logPnb) - logSumExp(a[1].logPb, a[1].logPnb))
        .slice(0, beamWidth),
    );
  }

  const raw = [...beamMap.entries()]
    .map(([text, { logPb, logPnb }]) => ({ text: text.trim(), rawProb: Math.exp(logSumExp(logPb, logPnb)) }))
    .sort((a, b) => b.rawProb - a.rawProb);

  const total = raw.reduce((s, b) => s + b.rawProb, 0) || 1;
  return raw.map(b => ({ text: b.text, prob: b.rawProb / total }));
}

// ── Transcript with word boundaries ──────────────────────────────────────────

/**
 * Standard CTC greedy decode, including ▁ as spaces.
 * Returns (text, charFrames) where charFrames[i] is the frame index of text[i].
 */
function ctcGreedyDecode(frameProbs: Float32Array[]): { text: string; charFrames: number[] } {
  let text = '';
  const charFrames: number[] = [];
  let lastId = -1;
  for (let f = 0; f < frameProbs.length; f++) {
    const probs = frameProbs[f];
    let maxId = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxId]) maxId = i;
    if (maxId === BLANK_ID) { lastId = -1; continue; }
    if (maxId === lastId) continue;
    lastId = maxId;
    const sym = ID_TO_TOKEN[maxId];
    if (!sym || sym.startsWith('<')) continue;
    if (sym === '▁') {
      // Only add a space if there isn't already one at the end.
      if (text.length > 0 && !text.endsWith(' ')) { text += ' '; charFrames.push(f); }
    } else {
      text += sym;
      charFrames.push(f);
    }
  }
  return { text: text.trim(), charFrames: text.trimStart() === text ? charFrames : charFrames.slice(text.length - text.trimStart().length) };
}

/**
 * If the greedy text has no spaces, insert them at blank-dominant gaps
 * between consecutive emitted characters (≥ GAP_FRAMES consecutive frames
 * where blank probability > 0.5).
 */
function insertGapSpaces(frameProbs: Float32Array[], text: string, charFrames: number[]): string {
  if (text.includes(' ')) return text;
  const GAP_FRAMES = 5; // ~200ms at 40ms/frame
  let result = '';
  for (let i = 0; i < text.length; i++) {
    if (i > 0 && charFrames[i] !== undefined && charFrames[i - 1] !== undefined) {
      let blankRun = 0;
      for (let f = charFrames[i - 1] + 1; f < charFrames[i]; f++) {
        if (frameProbs[f]?.[BLANK_ID] > 0.5) blankRun++;
      }
      if (blankRun >= GAP_FRAMES) result += ' ';
    }
    result += text[i];
  }
  return result;
}

/**
 * Build the best transcript string: greedy decode (which respects ▁ tokens)
 * with blank-gap spaces as a fallback when ▁ is never the argmax.
 */
function buildTranscript(frameProbs: Float32Array[]): string {
  const { text, charFrames } = ctcGreedyDecode(frameProbs);
  return insertGapSpaces(frameProbs, text, charFrames);
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (ev: MessageEvent<WorkerInMsg>) => {
  const msg = ev.data;

  if (msg.type === 'load') {
    try {
      const model = MODELS.find(m => m.id === msg.modelId);
      if (!model) throw new Error(`Unknown model: ${msg.modelId}`);

      if (loadedModelId === msg.modelId && session) {
        self.postMessage({ type: 'model-ready', modelId: msg.modelId } satisfies WorkerOutMsg);
        return;
      }

      const url = modelUrl(model);
      let modelData = await getModelFromCache(url);

      if (!modelData) {
        modelData = await downloadWithProgress(url, (pct, mbReceived, mbTotal) => {
          self.postMessage({ type: 'download-progress', pct, mbReceived, mbTotal } satisfies WorkerOutMsg);
        });
        await saveModelToCache(url, modelData);
      }

      session = await ort.InferenceSession.create(modelData, { executionProviders: ['wasm'] });
      loadedModelId = msg.modelId;
      self.postMessage({ type: 'model-ready', modelId: msg.modelId } satisfies WorkerOutMsg);
    } catch (e) {
      self.postMessage({ type: 'error', message: String(e) } satisfies WorkerOutMsg);
    }
  }

  if (msg.type === 'cancel') {
    cancelled = true;
    return;
  }

  if (msg.type === 'infer') {
    cancelled = false;
    const TIMEOUT_MS = 30_000;
    const timeoutId = setTimeout(() => {
      cancelled = true;
      self.postMessage({ type: 'error', message: 'Inference timed out after 30 s. Try a shorter recording or a smaller model.' } satisfies WorkerOutMsg);
    }, TIMEOUT_MS);

    try {
      if (!session) throw new Error('Model not loaded');

      const { features, numFrames } = extractFbank(msg.audio);
      if (cancelled) return;
      const inputTensor = new ort.Tensor('float32', features, [1, numFrames, 80]);
      const lensTensor  = new ort.Tensor('int64', new BigInt64Array([BigInt(numFrames)]), [1]);
      const outputs     = await session.run({ x: inputTensor, x_lens: lensTensor });
      if (cancelled) return;
      const raw         = outputs[session.outputNames[0]];

      const [, numCtcFrames, vocabSize] = raw.dims as number[];
      const data = raw.data as Float32Array;

      const frameProbs: Float32Array[] = [];
      const framesOut: FrameOut[] = [];

      for (let f = 0; f < numCtcFrames; f++) {
        const logits = data.slice(f * vocabSize, (f + 1) * vocabSize);
        const probs = softmax(logits);
        frameProbs.push(probs);

        // Top-5 tokens
        const top5 = Array.from(probs)
          .map((prob, id) => {
            const sym = ID_TO_TOKEN[id];
            if (sym === undefined) console.warn(`Unknown token id ${id} in vocab`);
            return { sym: sym ?? `?${id}`, prob };
          })
          .filter(t => !t.sym.startsWith('<') && t.sym !== '▁')
          .sort((a, b) => b.prob - a.prob)
          .slice(0, 5);

        framesOut.push({ top5 });
      }

      if (cancelled) return;
      const beams      = ctcBeamSearch(frameProbs);
      const transcript = buildTranscript(frameProbs);
      self.postMessage({ type: 'result', frames: framesOut, beams, transcript } satisfies WorkerOutMsg);
    } catch (e) {
      if (!cancelled) self.postMessage({ type: 'error', message: String(e) } satisfies WorkerOutMsg);
    } finally {
      clearTimeout(timeoutId);
    }
  }
};
