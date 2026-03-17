import type { FrameOut } from './types.js';

/** Returns true if the frame has a meaningful non-blank prediction. */
export function isActiveFrame(f: FrameOut): boolean {
  return f.top5.length > 0 && f.top5[0].prob >= 0.05;
}

export interface ActiveRange {
  first: number;
  last: number;
}

/**
 * Returns the index range [first, last] of active frames (inclusive),
 * trimming blank/low-signal frames from both ends.
 * Returns null if no active frame exists.
 */
export function findActiveRange(frames: FrameOut[]): ActiveRange | null {
  let first = -1;
  let last  = -1;
  for (let i = 0; i < frames.length; i++) {
    if (isActiveFrame(frames[i])) {
      if (first === -1) first = i;
      last = i;
    }
  }
  return first === -1 ? null : { first, last };
}
