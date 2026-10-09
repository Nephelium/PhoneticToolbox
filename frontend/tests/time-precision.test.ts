import {test} from 'node:test';
import assert from 'node:assert/strict';
import {timeText,limitTimeInput} from '../src/design/time-precision.ts';

test('time fields cap seconds at five and milliseconds at two decimals',()=>{
 for(const [value,seconds,ms] of [[14.985544858,'14.98554','14.99'],[16.364117232,'16.36412','16.36'],[1.378572,'1.37857','1.38'],[156.92141,'156.92141','156.92'],[-.0000001,'0','0'],[1e-5,'0.00001','0'],[1.999999,'2','2']] as const){
  assert.equal(timeText(value,'s'),seconds);assert.equal(timeText(value,'ms'),ms);
 }
});
test('editing keeps partial decimals, emptiness and allowed zeros, limits paste and exponent input',()=>{
 for(const value of ['', '-', '1.', '1.23000'])assert.equal(limitTimeInput(value,'s'),value);
 assert.equal(limitTimeInput('1.2300001','s'),'1.23');
 assert.equal(limitTimeInput('-156.92141','ms'),'-156.92');
 assert.equal(limitTimeInput('1.234567e0','s'),'1.23457');
 assert.equal(timeText('','ms'),'');assert.equal(timeText('invalid','s'),'invalid');
});
