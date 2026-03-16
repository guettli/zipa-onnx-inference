// Shared types and model list used by both main thread and worker.

export interface ModelInfo {
  id: string;
  label: string;
  hfRepo: string;
  file: string;
  sizeMb: number;
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
  | { type: 'load';  modelId: string }
  | { type: 'infer'; audio: Float32Array };

export type WorkerOutMsg =
  | { type: 'download-progress'; pct: number; mbReceived: number; mbTotal: number }
  | { type: 'model-ready'; modelId: string }
  | { type: 'result'; frames: FrameOut[]; beams: BeamOut[] }
  | { type: 'error'; message: string };

export interface FrameOut {
  top5: Array<{ sym: string; prob: number }>;
}

export interface BeamOut {
  text: string;
  prob: number;
}
