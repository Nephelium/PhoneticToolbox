import test from 'node:test';
import assert from 'node:assert/strict';
import {setLocale,messageRef} from '../src/i18n/index.ts';
import {responseError,uiErrorMessage} from '../src/modules/module-localization.ts';

test('parameter bridge errors retain their descriptor through live locale changes',async()=>{
 await setLocale('zh-CN');
 const raw='参数文件结构不受支持或包含公式。';
 const error=responseError(raw),notice=messageRef('');
 notice.value=uiErrorMessage(error);
 assert.equal(notice.value,raw);
 await setLocale('en');
 assert.equal(notice.value,'The parameter file structure is unsupported or contains formulas.');
 assert.equal(error.message,notice.value);
 await setLocale('ja');
 assert.equal(notice.value,'パラメーターファイルの構造に対応していないか、数式が含まれています。');
 assert.equal(error.message,notice.value);
 assert.equal(responseError('unrecognised diagnostic 123').message,'unrecognised diagnostic 123');
 await setLocale('zh-CN');
});
