/**
 * Computes phonetic neighbourhood scores between IPA symbols.
 *
 * Strategy: parse each symbol's English name into a set of phonetic features,
 * then score pairs by Jaccard similarity of those feature sets.
 *
 * Vowel features:  type:vowel, height:*, backness:*, round:*
 * Consonant features: type:consonant, voicing:*, place:*, manner:*
 *
 * Jaccard(A,B) = |A∩B| / |A∪B|  →  0 = nothing in common, 1 = identical
 */

type IpaEntry = { name: string; desc: string; url: string } | null;

function extractFeatures(name: string): Set<string> {
  const n = name.toLowerCase();
  const f = new Set<string>();

  const isVowel = n.includes('vowel');

  if (isVowel) {
    f.add('type:vowel');

    // Height (most specific first)
    if      (n.includes('near-close'))  f.add('height:near-close');
    else if (n.includes('close-mid'))   f.add('height:close-mid');
    else if (n.includes('near-open'))   f.add('height:near-open');
    else if (n.includes('open-mid'))    f.add('height:open-mid');
    else if (/\bclose\b/.test(n))       f.add('height:close');
    else if (/\bopen\b/.test(n))        f.add('height:open');
    else if (/\bmid\b/.test(n))         f.add('height:mid');

    // Backness
    if      (n.includes('near-front'))  f.add('backness:near-front');
    else if (n.includes('near-back'))   f.add('backness:near-back');
    else if (n.includes('front'))       f.add('backness:front');
    else if (n.includes('central'))     f.add('backness:central');
    else if (n.includes('back'))        f.add('backness:back');

    // Roundedness
    if      (n.includes('unrounded'))   f.add('round:unrounded');
    else if (n.includes('rounded'))     f.add('round:rounded');

  } else {
    f.add('type:consonant');

    // Voicing
    if      (n.includes('voiceless'))   f.add('voicing:voiceless');
    else if (n.includes('voiced'))      f.add('voicing:voiced');

    // Place of articulation (most specific first to avoid substring matches)
    if      (n.includes('alveolo-palatal'))               f.add('place:alveolo-palatal');
    else if (n.includes('postalveolar'))                  f.add('place:postalveolar');
    else if (n.includes('retroflex'))                     f.add('place:retroflex');
    else if (n.includes('labial\u2013velar') ||
             n.includes('labial-velar'))                  f.add('place:labial-velar');
    else if (n.includes('labial\u2013palatal') ||
             n.includes('labial-palatal'))                f.add('place:labial-palatal');
    else if (n.includes('labiodental'))                   f.add('place:labiodental');
    else if (n.includes('bilabial'))                      f.add('place:bilabial');
    else if (n.includes('dental'))                        f.add('place:dental');
    else if (n.includes('alveolar'))                      f.add('place:alveolar');
    else if (n.includes('palatal'))                       f.add('place:palatal');
    else if (n.includes('velar'))                         f.add('place:velar');
    else if (n.includes('uvular'))                        f.add('place:uvular');
    else if (n.includes('pharyngeal'))                    f.add('place:pharyngeal');
    else if (n.includes('epiglottal'))                    f.add('place:epiglottal');
    else if (n.includes('glottal'))                       f.add('place:glottal');

    // Manner of articulation (most specific first)
    if      (n.includes('lateral fricative'))             f.add('manner:lateral-fricative');
    else if (n.includes('lateral approximant'))           f.add('manner:lateral-approximant');
    else if (n.includes('lateral flap'))                  f.add('manner:lateral-flap');
    else if (n.includes('lateral'))                       f.add('manner:lateral');
    else if (n.includes('implosive'))                     f.add('manner:implosive');
    else if (n.includes('click'))                         f.add('manner:click');
    else if (n.includes('trill'))                         f.add('manner:trill');
    else if (n.includes('flap') || n.includes('tap'))    f.add('manner:flap');
    else if (n.includes('approximant'))                   f.add('manner:approximant');
    else if (n.includes('fricative'))                     f.add('manner:fricative');
    else if (n.includes('nasal'))                         f.add('manner:nasal');
    else if (n.includes('plosive') || n.includes('stop')) f.add('manner:plosive');
  }

  return f;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  let intersection = 0;
  for (const x of a) if (b.has(x)) intersection++;
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

export type Neighbour = { sym: string; score: number };

/**
 * For every symbol in `descs` (that has a non-null entry), compute the top-N
 * phonetically nearest neighbours by Jaccard feature similarity.
 */
export function computeAllNeighbours(
  descs: Record<string, IpaEntry>,
  topN = 5,
): Record<string, Neighbour[]> {
  const featureMap = new Map<string, Set<string>>();
  for (const [sym, entry] of Object.entries(descs)) {
    if (!entry) continue;
    const feats = extractFeatures(entry.name);
    if (feats.size > 1) featureMap.set(sym, feats); // >1 means we got real features beyond just type:*
  }

  const result: Record<string, Neighbour[]> = {};
  for (const [sym, feats] of featureMap) {
    const scores: Neighbour[] = [];
    for (const [other, otherFeats] of featureMap) {
      if (other === sym) continue;
      const score = jaccard(feats, otherFeats);
      if (score > 0) scores.push({ sym: other, score });
    }
    scores.sort((a, b) => b.score - a.score);
    result[sym] = scores.slice(0, topN);
  }
  return result;
}

/** Stable element id for an IPA symbol, based on its Unicode code points. */
export function symAnchorId(sym: string): string {
  return 'ipa-sym-' + Array.from(sym).map(c => c.codePointAt(0)!.toString(16)).join('');
}
