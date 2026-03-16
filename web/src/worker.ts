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
import { MODELS } from './types.js';

// ── Option A: single-threaded, no SharedArrayBuffer needed ────────────────────
ort.env.wasm.numThreads = 1;
// Load WASM files from CDN so we don't have to bundle them.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/';

// ── State ─────────────────────────────────────────────────────────────────────

let session: ort.InferenceSession | null = null;
let loadedModelId: string | null = null;

// ── Helpers ───────────────────────────────────────────────────────────────────

function modelUrl(m: ModelInfo) {
  return `https://huggingface.co/${m.hfRepo}/resolve/main/${m.file}`;
}

/** Download with resumable partial saves every 5 MB. */
async function downloadWithProgress(url: string, onProgress: (pct: number, mb: number, total: number) => void): Promise<ArrayBuffer> {
  const partial = await getPartialDownload(url);
  const headers: Record<string, string> = {};
  let received = partial ? partial.data.byteLength : 0;
  if (partial) headers['Range'] = `bytes=${received}-`;

  const resp = await fetch(url, { headers });
  if (!resp.ok && resp.status !== 206) throw new Error(`HTTP ${resp.status}`);

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

  if (msg.type === 'infer') {
    try {
      if (!session) throw new Error('Model not loaded');

      const { features, numFrames } = extractFbank(msg.audio);
      const inputTensor = new ort.Tensor('float32', features, [1, numFrames, 80]);
      const lensTensor  = new ort.Tensor('int64', new BigInt64Array([BigInt(numFrames)]), [1]);
      const outputs     = await session.run({ x: inputTensor, x_lens: lensTensor });
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
          .map((prob, id) => ({ sym: ID_TO_TOKEN[id] ?? `?${id}`, prob }))
          .filter(t => !t.sym.startsWith('<') && t.sym !== '▁')
          .sort((a, b) => b.prob - a.prob)
          .slice(0, 5);

        framesOut.push({ top5 });
      }

      const beams = ctcBeamSearch(frameProbs);
      self.postMessage({ type: 'result', frames: framesOut, beams } satisfies WorkerOutMsg);
    } catch (e) {
      self.postMessage({ type: 'error', message: String(e) } satisfies WorkerOutMsg);
    }
  }
};
