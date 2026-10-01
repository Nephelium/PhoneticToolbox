import test from 'node:test';
import assert from 'node:assert/strict';
import {fitPanels,layoutKey,panelWidth} from '../src/layout/panelWidths.ts';
const limit={min:180,max:520,initial:220};
test('P04 malformed saved widths fall back; finite widths stay within reachable limits',()=>{
 for(const bad of [null,undefined,'300',{},Infinity,NaN])assert.equal(panelWidth(bad,limit),220);
 assert.equal(panelWidth(-100,limit),180);assert.equal(panelWidth(900,limit),520);assert.equal(panelWidth(310.4,limit),310);
});
test('P04 small windows share available space without mutating preferred widths',()=>{
 const wanted=[420,400];const fitted=fitPanels(wanted,[180,200],600);
 assert.equal(fitted[0]+fitted[1],600);assert(fitted[0]>=180&&fitted[1]>=200);assert.deepEqual(wanted,[420,400]);
 assert.deepEqual(fitPanels(wanted,[180,200],900),wanted);assert.deepEqual(fitPanels(wanted,[180,200],200),[180,200]);
});
test('P04 widths are module-specific, account-isolated and reusable across projects',()=>{
 assert.equal(layoutKey('local:M01'),'layout.panels.local:M01');assert.notEqual(layoutKey('M01'),layoutKey('M02'));
 assert.equal(layoutKey('server:alice:project-1:M02'),layoutKey('server:alice:project-2:M02'));
 assert.notEqual(layoutKey('server:alice:p:M02'),layoutKey('server:bob:p:M02'));
});
