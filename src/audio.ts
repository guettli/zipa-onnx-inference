import { spawnSync } from 'child_process';

/**
 * Load an audio file and return a Float32Array of mono PCM samples at 16 kHz.
 * Requires ffmpeg in PATH.
 */
export function loadAudio(filePath: string): Float32Array {
  const result = spawnSync('ffmpeg', [
    '-i', filePath,
    '-f', 'f32le',   // output format: 32-bit float little-endian
    '-ar', '16000',  // resample to 16 kHz
    '-ac', '1',      // mono
    'pipe:1',        // write to stdout
  ], {
    maxBuffer: 256 * 1024 * 1024, // 256 MB
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  if (result.error) {
    if ((result.error as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new Error('ffmpeg not found in PATH. Install it (e.g. via flake.nix or apt install ffmpeg).');
    }
    throw result.error;
  }
  if (result.status !== 0) {
    const stderr = result.stderr?.toString() ?? '';
    throw new Error(`ffmpeg exited with code ${result.status}:\n${stderr}`);
  }

  const buf = result.stdout as Buffer;
  const samples = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  // Copy to detached buffer so we own it
  return Float32Array.from(samples);
}

/** Compute RMS energy of a slice of audio, returned as a value in [0, 1]. */
export function rmsEnergy(audio: Float32Array, start: number, end: number): number {
  const len = Math.min(end, audio.length) - start;
  if (len <= 0) return 0;
  let sum = 0;
  for (let i = start; i < start + len; i++) {
    sum += audio[i] * audio[i];
  }
  return Math.sqrt(sum / len);
}

/**
 * Compute per-CTC-frame RMS energies (one per output frame).
 * Maps CTC output frames linearly back onto the audio signal.
 */
export function computeFrameEnergies(audio: Float32Array, numCtcFrames: number): Float32Array {
  const energies = new Float32Array(numCtcFrames);
  const maxEnergy = findMaxRms(audio, numCtcFrames);
  for (let i = 0; i < numCtcFrames; i++) {
    const start = Math.floor((i / numCtcFrames) * audio.length);
    const end   = Math.floor(((i + 1) / numCtcFrames) * audio.length);
    const e = rmsEnergy(audio, start, end);
    energies[i] = maxEnergy > 0 ? e / maxEnergy : 0;
  }
  return energies;
}

function findMaxRms(audio: Float32Array, numCtcFrames: number): number {
  let max = 0;
  for (let i = 0; i < numCtcFrames; i++) {
    const start = Math.floor((i / numCtcFrames) * audio.length);
    const end   = Math.floor(((i + 1) / numCtcFrames) * audio.length);
    const e = rmsEnergy(audio, start, end);
    if (e > max) max = e;
  }
  return max;
}
