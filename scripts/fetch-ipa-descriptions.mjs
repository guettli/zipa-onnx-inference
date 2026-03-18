#!/usr/bin/env node
// Fetches Wikipedia summaries for all IPA vocabulary symbols and writes
// web/src/ipa-descriptions.json.  Run once; commit the result.
//
// Usage:  node scripts/fetch-ipa-descriptions.mjs

import { writeFileSync } from 'fs';

// All tokens from src/tokens.ts
const TOKENS = [
  '<blk>', '<sos/eos>', '<unk>', '▁',
  'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
  'æ','ç','ð','ø','ħ','ŋ','œ',
  'ǀ','ǁ','ǂ','ǃ',
  'ɐ','ɑ','ɒ','ɓ','ɔ','ɕ','ɖ','ɗ','ɘ','ə','ɛ','ɜ','ɞ','ɟ','ɠ','ɢ','ɣ','ɤ','ɥ','ɦ','ɧ','ɨ','ɪ',
  'ɬ','ɭ','ɮ','ɯ','ɰ','ɱ','ɲ','ɳ','ɴ','ɵ','ɶ','ɸ','ɹ','ɺ','ɻ','ɽ','ɾ',
  'ʀ','ʁ','ʂ','ʃ','ʄ','ʈ','ʉ','ʊ','ʋ','ʌ','ʍ','ʎ','ʏ','ʐ','ʑ','ʒ','ʔ','ʕ','ʘ','ʙ','ʛ','ʜ','ʝ','ʟ','ʡ','ʢ',
  'ʰ','ʲ','ʷ','ʼ','ː','˞','ˠ','ˤ',
  '̃','̚','̥','̩','̪','̴','̺',
  'β','θ','χ','ᶑ','ⱱ',
];

// Map symbol → Wikipedia article title (English)
// null = internal CTC token, no phonetics article
const WIKI_TITLE = {
  '<blk>':     null,
  '<sos/eos>': null,
  '<unk>':     null,
  '▁':         'Word_boundary',

  // Basic Latin IPA usage
  'a': 'Open_front_unrounded_vowel',
  'b': 'Voiced_bilabial_stop',
  'c': 'Voiceless_palatal_stop',
  'd': 'Voiced_alveolar_stop',
  'e': 'Close-mid_front_unrounded_vowel',
  'f': 'Voiceless_labiodental_fricative',
  'g': 'Voiced_velar_stop',
  'h': 'Voiceless_glottal_fricative',
  'i': 'Close_front_unrounded_vowel',
  'j': 'Voiced_palatal_approximant',
  'k': 'Voiceless_velar_stop',
  'l': 'Voiced_alveolar_lateral_approximant',
  'm': 'Voiced_bilabial_nasal',
  'n': 'Voiced_alveolar_nasal',
  'o': 'Close-mid_back_rounded_vowel',
  'p': 'Voiceless_bilabial_stop',
  'q': 'Voiceless_uvular_stop',
  'r': 'Voiced_alveolar_trill',
  's': 'Voiceless_alveolar_fricative',
  't': 'Voiceless_alveolar_stop',
  'u': 'Close_back_rounded_vowel',
  'v': 'Voiced_labiodental_fricative',
  'w': 'Voiced_labial%E2%80%93velar_approximant',
  'x': 'Voiceless_velar_fricative',
  'y': 'Close_front_rounded_vowel',
  'z': 'Voiced_alveolar_fricative',

  // Extended Latin / IPA letters
  'æ': 'Near-open_front_unrounded_vowel',
  'ç': 'Voiceless_palatal_fricative',
  'ð': 'Voiced_dental_fricative',
  'ø': 'Close-mid_front_rounded_vowel',
  'ħ': 'Voiceless_pharyngeal_fricative',
  'ŋ': 'Voiced_velar_nasal',
  'œ': 'Open-mid_front_rounded_vowel',

  // Clicks
  'ǀ': 'Dental_click',
  'ǁ': 'Lateral_click',
  'ǂ': 'Palatal_click',
  'ǃ': 'Retroflex_click',

  // ɐ–ɪ
  'ɐ': 'Near-open_central_vowel',
  'ɑ': 'Open_back_unrounded_vowel',
  'ɒ': 'Open_back_rounded_vowel',
  'ɓ': 'Voiced_bilabial_implosive',
  'ɔ': 'Open-mid_back_rounded_vowel',
  'ɕ': 'Voiceless_alveolo-palatal_fricative',
  'ɖ': 'Voiced_retroflex_stop',
  'ɗ': 'Voiced_alveolar_implosive',
  'ɘ': 'Close-mid_central_unrounded_vowel',
  'ə': 'Mid_central_vowel',
  'ɛ': 'Open-mid_front_unrounded_vowel',
  'ɜ': 'Open-mid_central_unrounded_vowel',
  'ɞ': 'Open-mid_central_rounded_vowel',
  'ɟ': 'Voiced_palatal_stop',
  'ɠ': 'Voiced_velar_implosive',
  'ɢ': 'Voiced_uvular_stop',
  'ɣ': 'Voiced_velar_fricative',
  'ɤ': 'Close-mid_back_unrounded_vowel',
  'ɥ': 'Voiced_labial%E2%80%93palatal_approximant',
  'ɦ': 'Voiced_glottal_fricative',
  'ɧ': 'Voiceless_postalveolar%E2%80%93velar_fricative',
  'ɨ': 'Close_central_unrounded_vowel',
  'ɪ': 'Near-close_near-front_unrounded_vowel',

  // ɬ–ɾ
  'ɬ': 'Voiceless_alveolar_lateral_fricative',
  'ɭ': 'Voiced_retroflex_lateral_approximant',
  'ɮ': 'Voiced_alveolar_lateral_fricative',
  'ɯ': 'Close_back_unrounded_vowel',
  'ɰ': 'Voiced_velar_approximant',
  'ɱ': 'Voiced_labiodental_nasal',
  'ɲ': 'Voiced_palatal_nasal',
  'ɳ': 'Voiced_retroflex_nasal',
  'ɴ': 'Voiced_uvular_nasal',
  'ɵ': 'Close-mid_central_rounded_vowel',
  'ɶ': 'Open_front_rounded_vowel',
  'ɸ': 'Voiceless_bilabial_fricative',
  'ɹ': 'Voiced_alveolar_approximant',
  'ɺ': 'Voiced_alveolar_lateral_flap',
  'ɻ': 'Voiced_retroflex_approximant',
  'ɽ': 'Voiced_retroflex_flap',
  'ɾ': 'Voiced_alveolar_flap',

  // ʀ–ʢ
  'ʀ': 'Voiced_uvular_trill',
  'ʁ': 'Voiced_uvular_fricative',
  'ʂ': 'Voiceless_retroflex_fricative',
  'ʃ': 'Voiceless_postalveolar_fricative',
  'ʄ': 'Voiced_palatal_implosive',
  'ʈ': 'Voiceless_retroflex_stop',
  'ʉ': 'Close_central_rounded_vowel',
  'ʊ': 'Near-close_near-back_vowel',
  'ʋ': 'Voiced_labiodental_approximant',
  'ʌ': 'Open-mid_back_unrounded_vowel',
  'ʍ': 'Voiceless_labial%E2%80%93velar_fricative',
  'ʎ': 'Voiced_palatal_lateral_approximant',
  'ʏ': 'Near-close_near-front_rounded_vowel',
  'ʐ': 'Voiced_retroflex_fricative',
  'ʑ': 'Voiced_alveolo-palatal_fricative',
  'ʒ': 'Voiced_postalveolar_fricative',
  'ʔ': 'Glottal_stop',
  'ʕ': 'Voiced_pharyngeal_fricative',
  'ʘ': 'Bilabial_click',
  'ʙ': 'Voiced_bilabial_trill',
  'ʛ': 'Voiced_uvular_implosive',
  'ʜ': 'Voiceless_epiglottal_fricative',
  'ʝ': 'Voiced_palatal_fricative',
  'ʟ': 'Voiced_velar_lateral_approximant',
  'ʡ': 'Voiceless_epiglottal_stop',
  'ʢ': 'Voiced_epiglottal_fricative',

  // Suprasegmentals / diacritics
  'ʰ': 'Aspirated_consonant',
  'ʲ': 'Palatalization_(phonetics)',
  'ʷ': 'Labialization',
  'ʼ': 'Ejective_consonant',
  'ː': 'Long_(phonetics)',
  '˞': 'Rhotacization_(phonetics)',
  'ˠ': 'Velarization',
  'ˤ': 'Pharyngealization',

  // Combining diacritics
  '̃':  'Nasalization',
  '̚':  'No_audible_release',
  '̥':  'Voicelessness',
  '̩':  'Syllabic_consonant',
  '̪':  'Dental_consonant',
  '̴':  'Velarization',
  '̺':  'Apical_consonant',

  // Greek letters used in IPA
  'β': 'Voiced_bilabial_fricative',
  'θ': 'Voiceless_dental_fricative',
  'χ': 'Voiceless_uvular_fricative',

  // Extensions
  'ᶑ': 'Voiced_retroflex_implosive',
  'ⱱ': 'Voiced_labiodental_flap',
};

const WIKI_REST = 'https://en.wikipedia.org/api/rest_v1/page/summary/';
const DELAY_MS = 150; // polite rate limit

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function fetchSummary(title) {
  const url = `${WIKI_REST}${title}`;
  const res = await fetch(url, {
    headers: {
      'User-Agent': 'zipa-onnx-inference/1.0 (https://github.com/zipa-project; educational)',
      'Accept': 'application/json',
    },
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} for "${title}"`);
  }
  const data = await res.json();
  return {
    name: data.title,
    desc: data.extract ?? '',
    url:  data.content_urls?.desktop?.page ?? `https://en.wikipedia.org/wiki/${title}`,
  };
}

const result = {};
let ok = 0, fail = 0;

for (const sym of TOKENS) {
  const title = WIKI_TITLE[sym];

  if (title === undefined) {
    console.warn(`⚠  No mapping for "${sym}" — skipping`);
    result[sym] = null;
    continue;
  }

  if (title === null) {
    // Internal token — no Wikipedia article
    result[sym] = null;
    continue;
  }

  try {
    await sleep(DELAY_MS);
    const info = await fetchSummary(title);
    result[sym] = info;
    console.log(`✓  ${sym.padEnd(4)} → ${info.name}`);
    ok++;
  } catch (e) {
    console.error(`✗  ${sym}: ${e.message}`);
    // Fallback: readable title + Wikipedia link, no description
    const decoded = decodeURIComponent(title).replace(/_/g, ' ');
    result[sym] = {
      name: decoded,
      desc: '',
      url:  `https://en.wikipedia.org/wiki/${title}`,
    };
    fail++;
  }
}

const outPath = new URL('../web/src/ipa-descriptions.json', import.meta.url).pathname;
writeFileSync(outPath, JSON.stringify(result, null, 2) + '\n');
console.log(`\nDone. ${ok} fetched, ${fail} failed. Written to ${outPath}`);
