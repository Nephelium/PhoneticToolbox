import test from 'node:test';
import assert from 'node:assert/strict';
import {setLocale} from '../src/i18n/index.ts';
import {diagnosticReason} from '../src/modules/phonology-induction/diagnostic-display.ts';

test('owned phonology parser reasons localize without changing source diagnostics or user text',async()=>{
 const source=['数据开始行之前','表头、空行或缺少字头 / IPA','只有调值或重音符号，缺少音段','疑似多个音节，请将每个音节放在独立记录中','含未知或替换符号，请核对原始 IPA'];
 const before=JSON.stringify(source),displays=[];
 for(const locale of ['zh-CN','en','ja'] as const){await setLocale(locale);displays.push(source.map(diagnosticReason));assert.equal(diagnosticReason('井井的原始记音 phoŋ1'), '井井的原始记音 phoŋ1');}
 assert.equal(JSON.stringify(source),before);
 for(let i=0;i<source.length;i++)assert.equal(new Set(displays.map(row=>row[i])).size,3);
 await setLocale('zh-CN');
});
