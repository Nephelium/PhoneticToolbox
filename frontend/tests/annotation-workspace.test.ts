import test from 'node:test';import assert from 'node:assert/strict';
import {editingNames,newIntervalTiers,editorSelection} from '../src/modules/annotation/layers.ts';
import {loadEditingGrid,parseGrid,serializeGrid} from '../src/modules/annotation/format.ts';
import {amplitudeLimit,amplitudeLabel} from '../src/platform/waveform.ts';
import {createEditor} from '../src/modules/annotation/editor.mjs';
test('M12-R2 empty input waits for explicit names; creation preserves other tiers and validates names',()=>{
 for(const text of [undefined,' \n','"ooTextFile short" "TextGrid" 0 2 <exists> 0']){
  const {grid}=loadEditingGrid(text,2);assert.deepEqual(grid.tiers,[]);assert.deepEqual(editingNames(grid),{word:'',phone:''});
  grid.tiers.push(...newIntervalTiers(grid,['音节','音素']));assert.deepEqual(parseGrid(serializeGrid(grid)).tiers.map(t=>t.name),['音节','音素']);
  for(const names of [[''],['same','same'],['音节'],['bad\nname']])assert.throws(()=>newIntervalTiers(grid,names));
 }
 assert.throws(()=>loadEditingGrid('bad nonempty grid',2));
});
test('M12-R2 roles select actual names and never substitute a remembered absent words layer',()=>{
 const grid={xmin:0,xmax:2,tiers:[{name:'events',points:[]},...newIntervalTiers({xmin:0,xmax:2,tiers:[]},['syllables','phones'])]};
 assert.deepEqual(editingNames(grid,{word:'words',phone:'phones'}),{word:'syllables',phone:'phones'});
 grid.tiers[1].name='自定义甲';grid.tiers[2].name='自定义乙';assert.deepEqual(editingNames(grid),{word:'自定义甲',phone:'自定义乙'});
 assert.deepEqual(editingNames({...grid,tiers:[grid.tiers[0]]}),{word:'',phone:''});
 assert.deepEqual(editingNames({...grid,tiers:[grid.tiers[2]]}),{word:'自定义乙',phone:''});
});
test('M12-R2 selection covers a group or an individual phone without changing boundaries',()=>{
 const e=createEditor();e.state.textgrid={xmin:0,xmax:2,tiers:[{name:'words',intervals:[{xmin:0,xmax:.2,text:''},{xmin:.2,xmax:.7,text:'a'},{xmin:.7,xmax:1.1,text:'b'}]},{name:'phones',intervals:[{xmin:.2,xmax:.35,text:'a'}]}]};
 e.state.selected={tier:'words',index:1};e.state.selectedIndices=[1,2];assert.deepEqual(editorSelection(e.state),[.2,1.1]);
 e.state.selected={tier:'phones',index:0};assert.deepEqual(editorSelection(e.state),[.2,.35]);
 e.state.selected=null;assert.equal(editorSelection(e.state),undefined);
});
test('M12-R2 visible amplitude reflects real peaks including quiet, negative, silent and >1 signals',()=>{
 const points:[number,number][]=[[-.03,.02],[-.01,.04]],copy=structuredClone(points);
 assert.equal(amplitudeLimit(points),.04);assert.deepEqual(points,copy);assert.equal(amplitudeLimit([[0,0]]),1);
 assert.equal(amplitudeLimit([[-2,.2]]),2);assert.equal(amplitudeLimit([[-.00002,.00001]]),.00002);
 assert.equal(amplitudeLabel(0),'0');assert.equal(amplitudeLabel(.00002),'2.00e-5');assert.equal(amplitudeLabel(-.04),'-0.04');
});
