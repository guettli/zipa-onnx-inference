// @ts-check
import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  eslint.configs.recommended,
  tseslint.configs.recommended,
  {
    rules: {
      // Warn on unused vars but allow underscore-prefixed to be ignored
      '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
      // Allow explicit any only when unavoidable
      '@typescript-eslint/no-explicit-any': 'warn',
    },
  },
  {
    // Scripts use top-level await which is fine
    files: ['scripts/**/*.ts', 'scripts/**/*.tsx'],
    rules: {},
  },
  {
    // .mjs scripts run in Node.js and use browser-like globals (fetch, console, etc.)
    files: ['scripts/**/*.mjs'],
    languageOptions: {
      globals: {
        console: 'readonly',
        fetch: 'readonly',
        URL: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        process: 'readonly',
      },
    },
  },
  {
    ignores: ['dist/**', 'node_modules/**', 'data/**'],
  },
);
