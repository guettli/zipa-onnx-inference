// Lightweight tests using only Node.js built-ins — run with: tsx src/frame-utils.test.ts
import assert from 'node:assert/strict';
import { isActiveFrame, findActiveRange } from './frame-utils.js';
import type { FrameOut } from './types.js';

let failed = 0;

function test(name: string, fn: () => void) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`  ✗ ${name}: ${msg}`);
    failed++;
  }
}

function frame(sym: string, prob: number): FrameOut {
  return { top5: [{ sym, prob }] };
}
const blank: FrameOut = { top5: [] };
const low: FrameOut   = { top5: [{ sym: 'a', prob: 0.04 }] };

// ── isActiveFrame ─────────────────────────────────────────────────────────────

test('isActiveFrame: active at exactly 0.05', () => {
  assert.ok(isActiveFrame(frame('a', 0.05)));
});

test('isActiveFrame: active above 0.05', () => {
  assert.ok(isActiveFrame(frame('b', 0.99)));
});

test('isActiveFrame: inactive when top5 is empty', () => {
  assert.ok(!isActiveFrame(blank));
});

test('isActiveFrame: inactive when prob < 0.05', () => {
  assert.ok(!isActiveFrame(low));
});

// ── findActiveRange ───────────────────────────────────────────────────────────

test('findActiveRange: null for empty array', () => {
  assert.equal(findActiveRange([]), null);
});

test('findActiveRange: null when all frames are blank', () => {
  assert.equal(findActiveRange([blank, blank, blank]), null);
});

test('findActiveRange: trims leading and trailing blanks', () => {
  const frames = [blank, blank, frame('a', 0.8), blank, frame('b', 0.7), blank];
  assert.deepEqual(findActiveRange(frames), { first: 2, last: 4 });
});

test('findActiveRange: single active frame in the middle', () => {
  assert.deepEqual(findActiveRange([blank, frame('x', 0.5), blank]), { first: 1, last: 1 });
});

test('findActiveRange: all frames active', () => {
  const frames = [frame('a', 0.9), frame('b', 0.8), frame('c', 0.7)];
  assert.deepEqual(findActiveRange(frames), { first: 0, last: 2 });
});

test('findActiveRange: low-prob frame is treated as blank', () => {
  const frames = [low, frame('a', 0.8), low];
  assert.deepEqual(findActiveRange(frames), { first: 1, last: 1 });
});

test('findActiveRange: blanks in the middle are included in range', () => {
  const frames = [frame('a', 0.8), blank, blank, frame('b', 0.9)];
  assert.deepEqual(findActiveRange(frames), { first: 0, last: 3 });
});

// ── Summary ───────────────────────────────────────────────────────────────────

if (failed > 0) {
  console.error(`\n${failed} test(s) failed`);
  process.exit(1);
} else {
  console.log('\nAll tests passed.');
}
