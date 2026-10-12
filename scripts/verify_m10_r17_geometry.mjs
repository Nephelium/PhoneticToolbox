import fs from 'node:fs';
import assert from 'node:assert/strict';
import {displayLips,connectedAirway,nasalOutlet,airwayReferencePaths,referenceNose} from '../frontend/public/vocal-tract/geometry.mjs';
import {registerNose} from '../frontend/public/vocal-tract/anatomy.mjs';
const states=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const nose=registerNose(JSON.parse(fs.readFileSync(new URL('../frontend/public/vocal-tract/assets/nasal.json',import.meta.url),'utf8'))),outlet=nasalOutlet(nose),ref=referenceNose(nose,outlet),rows=[];
for(const [name,raw] of Object.entries(states)){
  const s=displayLips(raw),start=performance.now(),full=airwayReferencePaths(connectedAirway(s,nose,outlet)),middle=performance.now(),reduced=airwayReferencePaths(connectedAirway(s,ref.nose,ref.outlet)),end=performance.now();
  assert.deepEqual(reduced,full,name+' identical welded reference section');
  rows.push({name,full_ms:middle-start,reduced_ms:end-middle,paths:full.length});
}
fs.writeFileSync(process.argv[3],JSON.stringify({passed:true,rows},null,2));
