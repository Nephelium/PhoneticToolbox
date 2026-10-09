import test from 'node:test';
import assert from 'node:assert/strict';
import {renderFormula} from '../src/manual/formula.ts';
test('scientific formulas render offline with accessible math and preserve the source',()=>{
  const value=renderFormula('f_0=\\frac{1}{T}',true);
  assert.equal(value.error,'');assert(value.html.includes('<math'));assert(value.html.includes('class="katex"'));
  assert.equal(value.source,'f_0=\\frac{1}{T}');
});
test('untrusted formulas cannot inject links or HTML and malformed content falls back',()=>{
  const blocked=renderFormula('\\href{javascript:alert(1)}{x}',true);
  assert(!blocked.html.includes('href="javascript:'));assert(!blocked.html.includes('<script>'));
  assert(renderFormula('\\frac{',true).error);assert(renderFormula('x'.repeat(12001),true).error);
});
