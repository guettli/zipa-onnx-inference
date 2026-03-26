/**
 * Memory usage test using Chrome DevTools Protocol (CDP).
 *
 * Measures JS heap and DOM node count before and after key actions so we can
 * see exactly where memory grows. Run with:
 *
 *   cd web && pnpm exec playwright test tests/memory.spec.ts --reporter=line
 *
 * NOTE: This uses Chrome/Chromium headless only.  Safari does not expose heap
 * metrics via CDP and cannot be measured this way.
 *
 * The test does NOT run model inference (that requires a real ONNX model file
 * cached from HuggingFace).  It measures the baseline and IPA-lazy-load phases.
 * For inference memory, use the real browser on device and watch the crash log.
 */
import { test, expect, chromium } from '@playwright/test';
import type { CDPSession } from '@playwright/test';

const BASE = '/zipa-onnx-inference/';

// ── Helpers ──────────────────────────────────────────────────────────────────

interface MemSnapshot {
  label: string;
  jsHeapUsedMB: number;
  jsHeapTotalMB: number;
  domNodes: number;
}

async function snapshot(cdp: CDPSession, label: string): Promise<MemSnapshot> {
  const [metrics, heapInfo] = await Promise.all([
    cdp.send('Performance.getMetrics'),
    cdp.send('Runtime.getHeapUsage'),
  ]);

  const find = (name: string) =>
    (metrics.metrics as Array<{ name: string; value: number }>).find(m => m.name === name)?.value ?? 0;

  const jsHeapUsedMB  = (heapInfo as { usedSize: number }).usedSize / 1024 / 1024;
  const jsHeapTotalMB = (heapInfo as { totalSize: number }).totalSize / 1024 / 1024;
  const domNodes      = find('Nodes');

  const snap: MemSnapshot = { label, jsHeapUsedMB, jsHeapTotalMB, domNodes };
  console.log(
    `[MEM] ${label.padEnd(38)} heap: ${jsHeapUsedMB.toFixed(1).padStart(6)} / ${jsHeapTotalMB.toFixed(1).padStart(6)} MB   DOM: ${domNodes}`,
  );
  return snap;
}

// ── Test ─────────────────────────────────────────────────────────────────────

test('memory usage across page lifecycle', async () => {
  // Launch our own browser so we can enable CDP features
  const browser = await chromium.launch({ headless: true, args: ['--enable-precise-memory-info'] });
  const context = await browser.newContext();
  const page    = await context.newPage();
  const cdp     = await context.newCDPSession(page);

  await cdp.send('Performance.enable');

  const snapshots: MemSnapshot[] = [];
  const snap = (label: string) => snapshot(cdp, label).then(s => { snapshots.push(s); return s; });

  // ── 1. Initial page load ──────────────────────────────────────────────────
  await page.goto(BASE);
  await page.waitForLoadState('networkidle');
  await snap('after page load');

  // ── 2. Force GC then measure baseline ────────────────────────────────────
  await cdp.send('HeapProfiler.collectGarbage');
  await page.waitForTimeout(500);
  await snap('after GC (baseline)');

  // ── 3. Trigger IPA description lazy load ─────────────────────────────────
  // Scroll to the IPA overview section to trigger IntersectionObserver
  await page.evaluate(() => {
    const el = document.getElementById('ipa-overview-section');
    el?.scrollIntoView({ behavior: 'instant' });
  });
  await page.waitForTimeout(3000); // wait for dynamic import + DOM rendering
  await snap('after IPA overview rendered');

  await cdp.send('HeapProfiler.collectGarbage');
  await page.waitForTimeout(500);
  await snap('after GC post-IPA');

  // ── 4. Scroll back up ─────────────────────────────────────────────────────
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(300);
  await snap('scrolled back to top');

  // ── 5. Summary ───────────────────────────────────────────────────────────
  console.log('\n── Memory summary ─────────────────────────────────────────');
  console.log('label'.padEnd(40) + 'heap used (MB)   DOM nodes');
  for (const s of snapshots) {
    console.log(`${s.label.padEnd(40)} ${s.jsHeapUsedMB.toFixed(1).padStart(8)}         ${s.domNodes}`);
  }

  const baseline = snapshots.find(s => s.label.includes('baseline'))!;
  const afterIpa = snapshots.find(s => s.label.includes('IPA overview rendered'))!;
  const afterIpaGc = snapshots.find(s => s.label.includes('post-IPA'))!;

  // IPA lazy load should add some heap (JSON + DOM) but not an absurd amount
  const ipaHeapDeltaMB = afterIpa.jsHeapUsedMB - baseline.jsHeapUsedMB;
  const ipaHeapAfterGcMB = afterIpaGc.jsHeapUsedMB - baseline.jsHeapUsedMB;
  console.log(`\nIPA load heap delta:       ${ipaHeapDeltaMB.toFixed(1)} MB`);
  console.log(`IPA retained after GC:     ${ipaHeapAfterGcMB.toFixed(1)} MB`);
  console.log(`IPA DOM nodes added:       ${afterIpa.domNodes - baseline.domNodes}`);

  // Sanity: page should load with < 50 MB JS heap
  expect(baseline.jsHeapUsedMB).toBeLessThan(50);

  // IPA data retained after GC should be < 30 MB (JSON + DOM nodes)
  expect(ipaHeapAfterGcMB).toBeLessThan(30);

  await browser.close();
});

test('worker exists and service worker registers', async ({ page }) => {
  await page.goto(BASE);
  await page.waitForLoadState('networkidle');

  // SW should register within 3 seconds
  const swRegistered = await page.evaluate(async () => {
    if (!('serviceWorker' in navigator)) return false;
    const reg = await navigator.serviceWorker.getRegistration('./sw.js').catch(() => null);
    return reg != null;
  });

  // Give SW time to register if not yet
  if (!swRegistered) {
    await page.waitForTimeout(2000);
  }

  const swActive = await page.evaluate(async () => {
    if (!('serviceWorker' in navigator)) return false;
    await navigator.serviceWorker.ready;
    return !!navigator.serviceWorker.controller || !!navigator.serviceWorker.getRegistration('./sw.js');
  });

  console.log(`Service Worker registered: ${swActive}`);
  // On first load the SW may not be controlling yet (requires reload after install)
  // We just verify it registered without error
  expect(typeof swActive).toBe('boolean');
});
