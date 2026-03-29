// Shared types and model list used by both main thread and worker.

export interface ModelInfo {
  id: string;
  label: string;
  hfRepo: string;
  file: string;
  sizeMb: number;
}

export function modelUrl(m: ModelInfo): string {
  return `https://huggingface.co/${m.hfRepo}/resolve/main/${m.file}`;
}

/** Same-origin proxy URL served by the Service Worker from Cache Storage. */
export function modelProxyUrl(m: ModelInfo): string {
  return `/zipa-onnx-inference/model-proxy/${encodeURIComponent(m.id)}`;
}

export const MODELS: ModelInfo[] = [
  { id: 'zipa-small-crctc-ns-700k/int8',  label: 'Small NS 700k  int8  ~65 MB (recommended)',  hfRepo: 'anyspeech/zipa-small-crctc-ns-700k',  file: 'model.int8.onnx', sizeMb: 65  },
  { id: 'zipa-small-crctc-300k/int8',     label: 'Small CTC 300k  int8  ~65 MB',               hfRepo: 'anyspeech/zipa-small-crctc-300k',     file: 'model.int8.onnx', sizeMb: 65  },
  { id: 'zipa-small-crctc-500k/int8',     label: 'Small CTC 500k  int8  ~65 MB',               hfRepo: 'anyspeech/zipa-small-crctc-500k',     file: 'model.int8.onnx', sizeMb: 65  },
  { id: 'zipa-large-crctc-ns-800k/int8',  label: 'Large NS 800k  int8  ~300 MB',               hfRepo: 'anyspeech/zipa-large-crctc-ns-800k',  file: 'model.int8.onnx', sizeMb: 300 },
  { id: 'zipa-small-crctc-ns-700k/fp32',  label: 'Small NS 700k  fp32  ~260 MB',               hfRepo: 'anyspeech/zipa-small-crctc-ns-700k',  file: 'model.onnx',      sizeMb: 260 },
  { id: 'zipa-small-crctc-300k/fp32',     label: 'Small CTC 300k  fp32  ~260 MB',              hfRepo: 'anyspeech/zipa-small-crctc-300k',     file: 'model.onnx',      sizeMb: 260 },
];

export type WorkerInMsg =
  | { type: 'load';   modelId: string }
  | { type: 'infer';  audio: Float32Array }
  | { type: 'cancel' };

export type WorkerOutMsg =
  | { type: 'download-progress'; pct: number; mbReceived: number; mbTotal: number }
  | { type: 'model-ready'; modelId: string; executionProvider: string }
  | { type: 'result'; frames: FrameOut[]; beams: BeamOut[]; transcript: string }
  | { type: 'error'; message: string };

// ── Whisper worker messages ───────────────────────────────────────────────────

export type WhisperWorkerInMsg =
  | { type: 'load' }
  | { type: 'infer'; audio: Float32Array; language: string }
  | { type: 'cancel' };

export type WhisperWorkerOutMsg =
  | { type: 'download-progress'; file: 'encoder' | 'decoder' | 'vocab'; pct: number; mbReceived: number; mbTotal: number }
  | { type: 'model-ready' }
  | { type: 'result'; transcript: string }
  | { type: 'error'; message: string };

/** Which models to run when audio is processed. */
export type InferenceMode = 'both' | 'zipa' | 'whisper';

export interface FrameOut {
  top5: Array<{ sym: string; prob: number }>;
}

export interface BeamOut {
  text: string;
  prob: number;
}
