/**
 * Web Worker: model download, ONNX inference, CTC beam search.
 *
 * Memory strategy
 * ───────────────
 * 1. Models are stored in the Cache Storage API (not IndexedDB).
 *    The Service Worker (sw.js) serves them at same-origin proxy URLs so that
 *    ort.InferenceSession.create(url) can fetch the model without our code ever
 *    holding a 65 MB ArrayBuffer in the JS heap simultaneously with the WASM copy.
 *
 * 2. Download is fully streaming via TransformStream → cache.put().
 *    Peak in-flight memory during download ≈ one network chunk (≤ 64 KB), not 65 MB.
 *
 * 3. Logits are processed frame-by-frame with two reusable buffers.
 *    No Float32Array[] of all frames is kept — only blankProbs[] (1 float/frame).
 *
 * 4. Single-threaded WASM (numThreads=1) — no SharedArrayBuffer required.
 */
import * as ort from 'onnxruntime-web';
import { extractFbank } from '../../src/fbank.js';
import { BLANK_ID, ID_TO_TOKEN } from '../../src/tokens.js';
import {
  getModelFromCache, deleteModelFromCache,
  getPartialDownload, savePartialDownload, clearPartialDownload,
} from './model-cache.js';
import type { ModelInfo, WorkerInMsg, WorkerOutMsg, FrameOut, BeamOut } from './types.js';
import { MODELS, modelUrl, modelProxyUrl } from './types.js';

// ── ONNX Runtime config ───────────────────────────────────────────────────────
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

// ── State ─────────────────────────────────────────────────────────────────────
let session: ort.InferenceSession | null = null;
let loadedModelId: string | null = null;
let cancelled = false;

const MODEL_CACHE_NAME = 'zipa-models-v1';

// ── Cache Storage helpers ─────────────────────────────────────────────────────

async function isModelInCacheStorage(proxyUrl: string): Promise<boolean> {
  if (!('caches' in self)) return false;
  const cache = await caches.open(MODEL_CACHE_NAME);
  return (await cache.match(proxyUrl)) != null;
}

/**
 * Stream-download a model from HuggingFace into Cache Storage.
 * Peak JS heap ≈ one network chunk (≤ 64 KB) — no full ArrayBuffer needed.
 *
 * Falls back to chunk-accumulation if TransformStream is unavailable (very old Safari).
 */
async function downloadModelToCache(
  hfUrl: string,
  proxyUrl: string,
  onProgress: (pct: number, mbReceived: number, mbTotal: number) => void,
): Promise<void> {
  let resp: Response;
  try {
    resp = await fetch(hfUrl);
  } catch {
    throw new Error('Network error — check your internet connection and try again.');
  }
  if (resp.status === 404) throw new Error('Model file not found (404). The model may have moved on Hugging Face.');
  if (resp.status === 429) throw new Error('Rate limited by Hugging Face (429). Please wait a moment and try again.');
  if (!resp.ok) throw new Error(`Download failed (HTTP ${resp.status}). Try reloading the page.`);

  const total = parseInt(resp.headers.get('content-length') ?? '0', 10);
  let received = 0;

  const cache = await caches.open(MODEL_CACHE_NAME);

  if (typeof TransformStream !== 'undefined' && resp.body) {
    // ── Streaming path: no ArrayBuffer in JS heap ─────────────────────────
    const ts = new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        received += chunk.byteLength;
        onProgress(total > 0 ? received / total : 0, received / 1e6, total / 1e6);
        controller.enqueue(chunk);
      },
    });

    await cache.put(
      proxyUrl,
      new Response(resp.body.pipeThrough(ts), {
        headers: {
          'Content-Type': 'application/octet-stream',
          ...(total > 0 ? { 'Content-Length': String(total) } : {}),
        },
      }),
    );
  } else {
    // ── Fallback: accumulate chunks then cache ────────────────────────────
    const reader = resp.body!.getReader();
    const chunks: Uint8Array[] = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      onProgress(total > 0 ? received / total : 0, received / 1e6, total / 1e6);
    }
    const merged = new Uint8Array(received);
    let offset = 0;
    for (const c of chunks) { merged.set(c, offset); offset += c.byteLength; }
    await cache.put(proxyUrl, new Response(merged.buffer, {
      headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(received) },
    }));
  }
}

/**
 * Migrate a legacy model from IndexedDB to Cache Storage, then delete from IndexedDB.
 * This is a one-time migration; subsequent loads use the Cache Storage path.
 */
async function migrateFromIndexedDB(hfUrl: string, proxyUrl: string): Promise<void> {
  const buf = await getModelFromCache(hfUrl); // legacy IndexedDB read
  if (!buf) return;
  const cache = await caches.open(MODEL_CACHE_NAME);
  await cache.put(proxyUrl, new Response(buf, {
    headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(buf.byteLength) },
  }));
  await deleteModelFromCache(hfUrl).catch(() => {}); // clean up IndexedDB
}

// ── Session creation ──────────────────────────────────────────────────────────

/** Always use single-threaded WASM — predictable, no SharedArrayBuffer required. */
async function createSession(uriOrBuffer: string | ArrayBuffer): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(uriOrBuffer as string, { executionProviders: ['wasm'] });
}

// ── Log-space helpers ─────────────────────────────────────────────────────────

const LOG_ZERO = -Infinity;

function logSumExp(a: number, b: number): number {
  if (!isFinite(a)) return b;
  if (!isFinite(b)) return a;
  const max = Math.max(a, b);
  return max + Math.log1p(Math.exp(Math.min(a, b) - max));
}

// ── Beam search (streaming, one frame at a time) ──────────────────────────────

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

function beamSearchStep(
  beamMap: Map<string, BeamState>,
  logP: Float64Array,
  beamWidth: number,
): Map<string, BeamState> {
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

  return new Map(
    [...next.entries()]
      .sort((a, b) => logSumExp(b[1].logPb, b[1].logPnb) - logSumExp(a[1].logPb, a[1].logPnb))
      .slice(0, beamWidth),
  );
}

function finalizeBeams(beamMap: Map<string, BeamState>): BeamOut[] {
  const raw = [...beamMap.entries()]
    .map(([text, { logPb, logPnb }]) => ({ text: text.trim(), rawProb: Math.exp(logSumExp(logPb, logPnb)) }))
    .sort((a, b) => b.rawProb - a.rawProb);
  const total = raw.reduce((s, b) => s + b.rawProb, 0) || 1;
  return raw.map(b => ({ text: b.text, prob: b.rawProb / total }));
}

// ── Gap-space insertion ───────────────────────────────────────────────────────

function insertGapSpaces(blankProbs: Float32Array, text: string, charFrames: number[]): string {
  if (text.includes(' ')) return text;
  const GAP_FRAMES = 5;
  let result = '';
  for (let i = 0; i < text.length; i++) {
    if (i > 0 && charFrames[i] !== undefined && charFrames[i - 1] !== undefined) {
      let blankRun = 0;
      for (let f = charFrames[i - 1] + 1; f < charFrames[i]; f++) {
        if (blankProbs[f] > 0.5) blankRun++;
      }
      if (blankRun >= GAP_FRAMES) result += ' ';
    }
    result += text[i];
  }
  return result;
}

// ── Top-5 token extraction (no large temporary arrays) ────────────────────────

function computeTop5(probs: Float32Array): Array<{ sym: string; prob: number }> {
  const ids: number[] = [];
  for (let id = 0; id < probs.length; id++) {
    const sym = ID_TO_TOKEN[id];
    if (!sym || sym.startsWith('<') || sym === '▁') continue;
    if (ids.length < 5) {
      ids.push(id);
      ids.sort((a, b) => probs[b] - probs[a]);
    } else if (probs[id] > probs[ids[4]]) {
      ids[4] = id;
      ids.sort((a, b) => probs[b] - probs[a]);
    }
  }
  return ids.map(id => ({ sym: ID_TO_TOKEN[id] ?? `?${id}`, prob: probs[id] }));
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (ev: MessageEvent<WorkerInMsg>) => {
  const msg = ev.data;

  // ── Load model ─────────────────────────────────────────────────────────────
  if (msg.type === 'load') {
    try {
      const model = MODELS.find(m => m.id === msg.modelId);
      if (!model) throw new Error(`Unknown model: ${msg.modelId}`);

      if (loadedModelId === msg.modelId && session) {
        self.postMessage({ type: 'model-ready', modelId: msg.modelId, executionProvider: 'wasm' } satisfies WorkerOutMsg);
        return;
      }

      const hfUrl    = modelUrl(model);
      const pUrl     = modelProxyUrl(model);
      const inCache  = await isModelInCacheStorage(pUrl);

      if (!inCache) {
        // Check for legacy IndexedDB entry and migrate silently
        const legacyBuf = await getModelFromCache(hfUrl);
        if (legacyBuf) {
          self.postMessage({ type: 'download-progress', pct: 0.5, mbReceived: 0, mbTotal: 0 } satisfies WorkerOutMsg);
          await migrateFromIndexedDB(hfUrl, pUrl);
          self.postMessage({ type: 'download-progress', pct: 1, mbReceived: 0, mbTotal: 0 } satisfies WorkerOutMsg);
        } else {
          // Fresh download — streaming into Cache Storage
          await downloadModelToCache(hfUrl, pUrl, (pct, mbReceived, mbTotal) => {
            self.postMessage({ type: 'download-progress', pct, mbReceived, mbTotal } satisfies WorkerOutMsg);
          });
        }
      }

      // ── Create ONNX session ───────────────────────────────────────────────
      // Prefer the proxy URL: ort fetches from the SW, which serves from Cache Storage.
      // This avoids holding a 65 MB ArrayBuffer in the JS heap while WASM loads.
      // Fall back to ArrayBuffer if the SW is not yet active (first install).
      let createdSession: ort.InferenceSession | null = null;
      try {
        createdSession = await createSession(pUrl);
      } catch {
        // SW not yet active — fall back to reading buffer from Cache Storage
        const cache = await caches.open(MODEL_CACHE_NAME);
        const cached = await cache.match(pUrl);
        if (!cached) throw new Error('Model not in cache — please reload and try again.');
        const buf = await cached.arrayBuffer();
        createdSession = await createSession(buf);
      }

      session = createdSession;
      loadedModelId = msg.modelId;
      self.postMessage({ type: 'model-ready', modelId: msg.modelId, executionProvider: 'wasm' } satisfies WorkerOutMsg);
    } catch (e) {
      self.postMessage({ type: 'error', message: String(e) } satisfies WorkerOutMsg);
    }
  }

  if (msg.type === 'cancel') {
    cancelled = true;
    return;
  }

  // ── Inference ──────────────────────────────────────────────────────────────
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

      // Dispose input tensors to free WASM allocations.
      inputTensor.dispose();
      lensTensor.dispose();

      if (cancelled) return;

      const raw = outputs[session.outputNames[0]];
      const [, numCtcFrames, vocabSize] = raw.dims as number[];
      const data = raw.data as Float32Array;

      // ── Streaming frame processing ────────────────────────────────────────
      // Two small reusable buffers instead of allocating Float32Array(vocabSize) per frame.
      // blankProbs (1 float/frame) is the only per-frame allocation kept long-term.
      const probsBuf   = new Float32Array(vocabSize);
      const logPBuf    = new Float64Array(vocabSize);
      const blankProbs = new Float32Array(numCtcFrames);

      const framesOut: FrameOut[] = [];

      let beamMap = new Map<string, BeamState>([
        ['', { logPb: 0, logPnb: LOG_ZERO, lastId: -1 }],
      ]);

      let greedyText = '';
      const greedyCharFrames: number[] = [];
      let lastGreedyId = -1;

      for (let f = 0; f < numCtcFrames; f++) {
        // In-place softmax directly into probsBuf — no data.slice() copy per frame
        const offset = f * vocabSize;
        let max = -Infinity;
        for (let i = 0; i < vocabSize; i++) { const v = data[offset + i]; if (v > max) max = v; }
        let sum = 0;
        for (let i = 0; i < vocabSize; i++) { probsBuf[i] = Math.exp(data[offset + i] - max); sum += probsBuf[i]; }
        for (let i = 0; i < vocabSize; i++) probsBuf[i] /= sum;

        for (let i = 0; i < vocabSize; i++) {
          logPBuf[i] = probsBuf[i] > 0 ? Math.log(probsBuf[i]) : LOG_ZERO;
        }

        blankProbs[f] = probsBuf[BLANK_ID];
        beamMap = beamSearchStep(beamMap, logPBuf, 10);

        // Greedy decode step
        let maxId = 0;
        for (let i = 1; i < vocabSize; i++) if (probsBuf[i] > probsBuf[maxId]) maxId = i;
        if (maxId === BLANK_ID) {
          lastGreedyId = -1;
        } else if (maxId !== lastGreedyId) {
          lastGreedyId = maxId;
          const sym = ID_TO_TOKEN[maxId];
          if (sym && !sym.startsWith('<')) {
            if (sym === '▁') {
              if (greedyText.length > 0 && !greedyText.endsWith(' ')) {
                greedyText += ' ';
                greedyCharFrames.push(f);
              }
            } else {
              greedyText += sym;
              greedyCharFrames.push(f);
            }
          }
        }

        framesOut.push({ top5: computeTop5(probsBuf) });
      }

      if (cancelled) return;

      const beams      = finalizeBeams(beamMap);
      const transcript = insertGapSpaces(blankProbs, greedyText.trim(), greedyCharFrames);
      self.postMessage({ type: 'result', frames: framesOut, beams, transcript } satisfies WorkerOutMsg);
    } catch (e) {
      if (!cancelled) self.postMessage({ type: 'error', message: String(e) } satisfies WorkerOutMsg);
    } finally {
      clearTimeout(timeoutId);
    }
  }
};
