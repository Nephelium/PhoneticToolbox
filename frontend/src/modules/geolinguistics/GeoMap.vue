<script setup lang="ts">
import {onMounted,onBeforeUnmount,watch,ref,nextTick,computed,useId} from 'vue';
import OlMap from 'ol/Map.js';import View from 'ol/View.js';import Feature from 'ol/Feature.js';
import VectorLayer from 'ol/layer/Vector.js';import VectorSource from 'ol/source/Vector.js';import GeoJSON from 'ol/format/GeoJSON.js';
import VectorImageLayer from 'ol/layer/VectorImage.js';
import Point from 'ol/geom/Point.js';import GeometryCollection from 'ol/geom/GeometryCollection.js';import type Geometry from 'ol/geom/Geometry.js';import {Fill,Stroke,Style,Circle,Text} from 'ol/style.js';
import {fromLonLat,toLonLat,transformExtent} from 'ol/proj.js';import DragBox from 'ol/interaction/DragBox.js';import Draw from 'ol/interaction/Draw.js';import {shiftKeyOnly} from 'ol/events/condition.js';
import {wrapX} from 'ol/coordinate.js';
import {defaults as controls,ScaleLine} from 'ol/control.js';import type {GeoPort} from '../../platform/geolinguistics.ts';
import type {MapPoint,ViewState} from './model.ts';import 'ol/ol.css';
import {wrapMapText} from './export.ts';
import {mapAttributionText,type MapAttribution} from './attribution.ts';
import {mapPaint,mapPaintPriority,contextBoundaryVisible,contextGeometryVisible} from './map-paint.ts';
import {projectedBaseFeatures,projectedContextFeatures,projectedOutlineFeatures} from './projected-base.ts';
import {mapHoverName,type HoverLayer} from './map-hover.ts';
import {activeBoundary,boundaryLevels,boundaryPaint,type BoundaryLevel,type MapExportOptions} from './map-options.ts';
const props=defineProps<{port:GeoPort;points:MapPoint[];view:ViewState;active:boolean;level:'auto'|'province'|'city'|'district';extraBoundaries?:BoundaryLevel[];polygonColors?:Record<string,string>;fit?:{extent:number[];nonce:number};drawMode?:'Polygon'|'LineString'|'point'|'';drawings?:unknown;labels?:boolean;attributions?:MapAttribution[]}>();
const emit=defineEmits<{site:[id:string];region:[id:string];selection:[ids:string[]];view:[state:ViewState];extent:[extent:number[]];addPoint:[lon:number,lat:number];draw:[geojson:object];error:[message:string]}>();
const target=ref<HTMLDivElement>(),loading=ref(false),hover=ref('');const format=new GeoJSON();
const attributionOpen=ref(false),attributionButton=ref<HTMLButtonElement>(),attributionId='map-sources-'+useId(),attributionStyle=ref<Record<string,string>>({});
const attributionHost=ref<HTMLElement|string>('body');
function fullscreenChanged(){const full=document.fullscreenElement;attributionHost.value=full?.contains(target.value!)?full as HTMLElement:'body';attributionOpen.value=false;}
const osmAttribution=computed(()=>props.attributions?.find(item=>item.id==='osm'));
function closeAttribution(){attributionOpen.value=false;attributionButton.value?.focus();}
function attributionKeydown(event:KeyboardEvent){if(event.key==='Escape'&&attributionOpen.value){event.stopPropagation();event.preventDefault();closeAttribution();}}
function positionAttribution(){if(!attributionOpen.value||!attributionButton.value)return;const rect=attributionButton.value.getBoundingClientRect(),width=Math.min(320,innerWidth-16),above=rect.top>innerHeight/2;attributionStyle.value={left:Math.max(8,Math.min(rect.right-width,innerWidth-width-8))+'px',width:width+'px',maxHeight:Math.max(0,Math.min(260,above?rect.top-12:innerHeight-rect.bottom-12))+'px',top:above?'auto':rect.bottom+4+'px',bottom:above?innerHeight-rect.top+4+'px':'auto'};}
function toggleAttribution(){attributionOpen.value=!attributionOpen.value;positionAttribution();}
let map:OlMap|undefined,observer:ResizeObserver|undefined,draw:Draw|undefined,disposed=false,baseReady=false,levelEpoch=0,lastMapError='';
const territory=new VectorSource(),admin=new VectorSource(),boundaries=new VectorSource(),points=new VectorSource(),sketch=new VectorSource();
const boundaryFeatures=new Map<BoundaryLevel,readonly Feature<Geometry>[]>(),boundaryData=new Map<BoundaryLevel,object>();
const boundaryStyles=new Map<BoundaryLevel,Style>();
for(const level of boundaryLevels){const p=boundaryPaint(level);boundaryStyles.set(level,new Style({stroke:new Stroke({color:p.color,width:p.width}),zIndex:3-boundaryLevels.indexOf(level)}));}
const hoverLayers:HoverLayer[]=[{source:points,kind:'points'},{source:sketch,kind:'sketch'},{source:territory,kind:'territory'},{source:admin,kind:'admin'}];
const contextStyles=new WeakMap<Geometry,Style[]>(),adminStyles=new Map<string,Style>();
let pointStyles=new WeakMap<Feature,Style>();
const labelWidths=new Map<string,number>();let maxLabelWidth=0,hoverFrame=0,hoverCoordinate:number[]|undefined;
let labelMeasure:CanvasRenderingContext2D|null|undefined;
function labelWidth(name:string){let width=labelWidths.get(name);if(width===undefined){labelMeasure??=document.createElement('canvas').getContext('2d');if(labelMeasure)labelMeasure.font='12px "Microsoft YaHei",sans-serif';width=labelMeasure?.measureText(name).width??name.length*12;labelWidths.set(name,width);}return width;}
function syncHoverLabels(){labelWidths.clear();maxLabelWidth=props.labels?props.points.reduce((width,p)=>Math.max(width,labelWidth(p.name)),0):0;}
function clearHover(){if(hoverFrame)cancelAnimationFrame(hoverFrame);hoverFrame=0;hoverCoordinate=undefined;hover.value='';}
function scheduleHover(coordinate:number[]){
 hoverCoordinate=coordinate;if(hoverFrame)return;
 hoverFrame=requestAnimationFrame(()=>{hoverFrame=0;const coordinate=hoverCoordinate,view=map?.getView();if(!coordinate||!view||disposed||!props.active||view.getInteracting()||view.getAnimating())return;hover.value=mapHoverName(hoverLayers,wrapX([...coordinate],view.getProjection()),view.getResolution()!,!!props.labels,labelWidth,maxLabelWidth);});
}
function territoryStyles(geometry:Geometry,role?:string,line?:string):Style[]{
 const cached=contextStyles.get(geometry);if(cached)return cached;
 if(geometry instanceof GeometryCollection){const styles=geometry.getGeometries().flatMap(child=>territoryStyles(child,role,line));contextStyles.set(geometry,styles);return styles;}
 if(!contextGeometryVisible(geometry.getType(),role))return [];
 const p=mapPaint(1,geometry.getType(),undefined,role,line);
 const styles=[new Style({geometry,zIndex:mapPaintPriority(geometry.getType(),role),fill:p.fill==='none'?undefined:new Fill({color:p.fill}),stroke:new Stroke({color:p.stroke,width:p.width,lineDash:p.dash})})];
 contextStyles.set(geometry,styles);return styles;
}
// Reuse a viewport image through native drag/zoom gestures; redraw vectors at
// the settled resolution. Survey points and editable drawings stay vectors.
const adminLayer=new VectorImageLayer({source:admin,imageRatio:1.25,style:f=>{
 const color=props.polygonColors?.[String(f.getId())],key=(lastLevel||currentLevel())+':'+(color??'');let style=adminStyles.get(key);
 if(!style){const p=mapPaint(2,'Polygon',color);style=new Style({fill:new Fill({color:p.fill}),stroke:new Stroke({color:p.stroke,width:boundaryPaint((lastLevel||currentLevel()) as BoundaryLevel).width})});adminStyles.set(key,style);}return style;
}});
const territoryLayer=new VectorImageLayer({source:territory,imageRatio:1.25,style:f=>f instanceof Feature&&f.getGeometry()&&contextBoundaryVisible(f.get('boundaryType'))?territoryStyles(f.getGeometry()!,f.get('renderRole'),f.get('line')):undefined});
const boundaryLayer=new VectorImageLayer({source:boundaries,imageRatio:1.25,visible:false,style:f=>boundaryStyles.get(f.get('boundaryLevel'))});
const sketchLayer=new VectorLayer({source:sketch,style:new Style({fill:new Fill({color:'#397da525'}),stroke:new Stroke({color:'#28577b',width:2,lineDash:[6,4]})})});
const pointLayer=new VectorLayer({source:points,style:f=>{
 if(!(f instanceof Feature))return;const cached=pointStyles.get(f);if(cached)return cached;
 const p=f.get('datum') as MapPoint,style=new Style({image:new Circle({radius:p.multi?5.5:4,fill:new Fill({color:p.unclassified?'#ffffff':p.color}),stroke:new Stroke({color:p.selected?'#172c40':p.incomplete?'#526c78':'#ffffff',width:p.selected||p.incomplete?1.5:1,lineDash:p.incomplete&&!p.unclassified?[2,2]:undefined})}),text:props.labels?new Text({text:p.name,font:'12px "Microsoft YaHei",sans-serif',offsetY:-13,fill:new Fill({color:'#203445'}),stroke:new Stroke({color:'#fff',width:3})}):undefined,zIndex:p.selected?2:1});
 pointStyles.set(f,style);return style;
}});
const layers=[adminLayer,territoryLayer,boundaryLayer,sketchLayer,pointLayer];
function read(data:object){return format.readFeatures(data,{dataProjection:'EPSG:4326',featureProjection:'EPSG:3857'});}
function syncPoints(){clearHover();syncHoverLabels();points.clear();points.addFeatures(props.points.map(p=>{const f=new Feature({geometry:new Point(fromLonLat([p.lon,p.lat])),datum:p});f.setId(p.id);return f;}));}
function currentLevel(){return activeBoundary(props.level,map?.getView().getZoom()??4);}
let lastLevel='',requestedLevel='',requestedBoundaries='';
async function readBoundary(level:BoundaryLevel){const cached=boundaryFeatures.get(level);if(cached)return cached;const data=await props.port.map(level);if(disposed)return [];const features=projectedBaseFeatures(data);boundaryFeatures.set(level,features);boundaryData.set(level,data);return features;}
async function syncLevel(){
 if(!map)return;const level=currentLevel(),extra=boundaryLevels.filter(value=>value!==level&&props.extraBoundaries?.includes(value)),key=[level,...extra].join('|');
 if(key===requestedBoundaries)return;requestedBoundaries=key;requestedLevel=level;const epoch=++levelEpoch;loading.value=true;lastMapError='';
 try{const loaded=await Promise.all([level,...extra].map(readBoundary));if(disposed||epoch!==levelEpoch)return;
  if(lastLevel!==level){lastLevel=level;admin.clear();admin.addFeatures([...loaded[0]]);adminStyles.clear();adminLayer.changed();}
  boundaries.clear();boundaries.addFeatures(extra.flatMap(level=>[...projectedOutlineFeatures(boundaryData.get(level)!,level)]));boundaryLayer.setVisible(extra.length>0);
 }catch(e){if(!disposed&&epoch===levelEpoch){requestedLevel='';requestedBoundaries='';lastMapError=(e as Error).message;emit('error',lastMapError);}}finally{if(epoch===levelEpoch)loading.value=false;}
}
async function prepareExport(options:MapExportOptions){await Promise.all(options.boundaries.map(readBoundary));if(disposed)throw Error('地图已关闭。');}
async function loadBase(){const border=await props.port.map('territory');if(disposed)return false;territory.clear();territory.addFeatures([...projectedContextFeatures(border)]);baseReady=true;return true;}
async function reload(){if(disposed||!map)throw Error('地图尚未载入。');if(!baseReady&&!await loadBase())throw Error('地图已关闭。');boundaryFeatures.clear();lastLevel='';requestedLevel='';requestedBoundaries='';await syncLevel();if(!lastLevel||lastLevel!==requestedLevel||lastLevel!==currentLevel())throw Error(lastMapError||'当前地图边界未能载入，请重试。');}
function syncDraw(){if(!map)return;if(draw)map.removeInteraction(draw);draw=undefined;if(props.drawMode==='Polygon'||props.drawMode==='LineString'){draw=new Draw({source:sketch,type:props.drawMode});draw.on('drawend',event=>{const items=[...sketch.getFeatures(),event.feature];emit('draw',format.writeFeaturesObject(items,{featureProjection:'EPSG:3857',dataProjection:'EPSG:4326'}));});map.addInteraction(draw);}}
function publishView(){clearHover();if(!map||!props.active)return;const v=map.getView(),center=toLonLat(v.getCenter()!);emit('view',{center:[center[0],center[1]],zoom:v.getZoom()??4});emit('extent',transformExtent(v.calculateExtent(map.getSize()),'EPSG:3857','EPSG:4326'));void syncLevel();}
onMounted(async()=>{document.addEventListener('fullscreenchange',fullscreenChanged);window.addEventListener('resize',positionAttribution);window.addEventListener('scroll',positionAttribution,true);map=new OlMap({target:target.value,layers,view:new View({center:fromLonLat(props.view.center),zoom:props.view.zoom,minZoom:1,maxZoom:16}),controls:controls({rotate:false,attribution:false}).extend([new ScaleLine({units:'metric'})])});target.value?.querySelector('.ol-scale-line')?.setAttribute('data-background-surface','');
 if(props.fit)map.getView().fit(transformExtent(props.fit.extent,'EPSG:4326','EPSG:3857'),{padding:[25,25,25,25],maxZoom:12});
 map.on('moveend',publishView);map.on('pointermove',e=>{if(e.dragging){clearHover();return;}scheduleHover(e.coordinate);});target.value?.addEventListener('pointerleave',clearHover);
 map.on('singleclick',e=>{if(props.drawMode==='point'){const p=toLonLat(e.coordinate);emit('addPoint',p[0],p[1]);return;}if(props.drawMode)return;let hit=false;map?.forEachFeatureAtPixel(e.pixel,f=>{emit('site',String(f.getId()));hit=true;return true;},{hitTolerance:6,layerFilter:layer=>layer===pointLayer});if(!hit)map?.forEachFeatureAtPixel(e.pixel,f=>{emit('region',String(f.getId()));return true;},{hitTolerance:1,layerFilter:layer=>layer===adminLayer});});
 const box=new DragBox({condition:shiftKeyOnly});box.on('boxend',()=>{const extent=box.getGeometry().getExtent();emit('selection',points.getFeaturesInExtent(extent).map(f=>String(f.getId())));});map.addInteraction(box);
 observer=new ResizeObserver(()=>{if(props.active)map?.updateSize();});observer.observe(target.value!);syncPoints();syncDraw();
 try{if(await loadBase()){await syncLevel();publishView();}}catch(e){if(!disposed)emit('error',(e as Error).message);}
});
watch(()=>props.points,syncPoints);watch(()=>props.labels,()=>{clearHover();syncHoverLabels();pointStyles=new WeakMap();pointLayer.changed();});watch(()=>props.level,()=>{clearHover();void syncLevel();});watch(()=>props.polygonColors,()=>{adminStyles.clear();adminLayer.changed();});
watch(()=>props.extraBoundaries,()=>{void syncLevel();},{deep:true});
watch(()=>props.view,v=>{const view=map?.getView();if(!view)return;const old=toLonLat(view.getCenter()!);if(Math.abs(old[0]-v.center[0])>1e-6||Math.abs(old[1]-v.center[1])>1e-6||Math.abs((view.getZoom()??4)-v.zoom)>1e-6){view.setCenter(fromLonLat(v.center));view.setZoom(v.zoom);}});
watch(()=>props.fit,fit=>{if(fit&&map)map.getView().fit(transformExtent(fit.extent,'EPSG:4326','EPSG:3857'),{padding:[35,35,35,35],maxZoom:12,duration:250});});
watch(()=>props.drawMode,syncDraw);watch(()=>props.drawings,value=>{sketch.clear();if(value)sketch.addFeatures(read(value as object));},{immediate:true});
watch(()=>props.active,async active=>{if(active){await nextTick();map?.updateSize();}else{clearHover();attributionOpen.value=false;}});
onBeforeUnmount(()=>{
 disposed=true;levelEpoch++;clearHover();adminStyles.clear();labelWidths.clear();labelMeasure=undefined;target.value?.removeEventListener('pointerleave',clearHover);document.removeEventListener('fullscreenchange',fullscreenChanged);window.removeEventListener('resize',positionAttribution);window.removeEventListener('scroll',positionAttribution,true);observer?.disconnect();observer=undefined;
 const ownedMap=map;map=undefined;
 if(ownedMap){const interactions=[...ownedMap.getInteractions().getArray()],ownedControls=[...ownedMap.getControls().getArray()];ownedMap.setTarget(undefined);ownedMap.setLayers([]);ownedMap.dispose();for(const item of [...interactions,...ownedControls])item.dispose();}
 draw=undefined;for(const layer of layers)layer.dispose();
 // A cached Feature can outlive this map. Detach this Source's listeners only.
 for(const source of [territory,admin,boundaries,points,sketch]){source.clear(true);source.dispose();}boundaryFeatures.clear();boundaryData.clear();boundaryStyles.clear();
});
const escape=(s:string)=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
function exportSvg(title:string,legend:{label:string;color:string}[],source:string,options?:MapExportOptions){
 if(props.attributions?.length)source+='。'+mapAttributionText(props.attributions);
 if(disposed||!map)throw Error('地图尚未载入。');
 if(!baseReady||loading.value||!lastLevel||requestedLevel!==lastLevel||currentLevel()!==lastLevel)throw Error('当前地图边界尚未完整载入，请等待载入或重新选择边界层级后导出。');
 if(options&&!options.legend)legend=[];
 if(options?.boundaries.some(level=>!boundaryFeatures.has(level)))throw Error('导出边界尚未载入，请重试。');
 const [viewWidth,viewHeight]=map.getSize()!;
 // Export at a readable page width even when four small panels share a window.
 const w=1200,factor=w/viewWidth,h=Math.round(viewHeight*factor);
 const measure=document.createElement('canvas').getContext('2d')!;
 const wrap=(text:string,width:number,size:number)=>{measure.font=`${size}px "PTB-Doulos","Microsoft YaHei",sans-serif`;return wrapMapText(text,width,s=>measure.measureText(s).width);};
 const titleLines=wrap(title,w-32,22),sourceLines=wrap(source,w-32,12),cols=Math.max(2,Math.floor(w/220));
 const legendLines=legend.map(v=>wrap(v.label,w/cols-38,14));
 const rowHeights=Array.from({length:Math.ceil(legend.length/cols)},(_,i)=>Math.max(...legendLines.slice(i*cols,(i+1)*cols).map(v=>v.length))*20+8);
 const legendTop=h+24+titleLines.length*28,footer=50+titleLines.length*28+rowHeights.reduce((a,b)=>a+b,0)+sourceLines.length*18;
 const xy=(c:number[])=>map!.getPixelFromCoordinate(c).map(n=>n*factor);const line=(coords:number[][])=>coords.map((c,i)=>{const p=xy(c);return `${i?'L':'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`;}).join('');
 const path=(g:any):string=>g.type==='Polygon'?g.coordinates.map((r:number[][])=>line(r)+'Z').join(''):g.type==='MultiPolygon'?g.coordinates.map((p:number[][][])=>p.map(r=>line(r)+'Z').join('')).join(''):g.type==='LineString'?line(g.coordinates):g.type==='MultiLineString'?g.coordinates.map(line).join(''):g.type==='GeometryCollection'?g.geometries.map(path).join(''):'';
 let s=`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h+footer}" viewBox="0 0 ${w} ${h+footer}"><rect width="100%" height="100%" fill="#fff"/><defs><clipPath id="frame"><rect width="${w}" height="${h}"/></clipPath></defs><g clip-path="url(#frame)"><rect width="${w}" height="${h}" fill="#e5eff2"/>`;
 const paintedPath=(g:any,i:number,color?:string,role?:string,legacyLine?:string):string=>{
  if(g.type==='GeometryCollection')return g.geometries.map((child:any)=>paintedPath(child,i,color,role,legacyLine)).join('');
  const p=mapPaint(i,g.type,color,role,legacyLine);
  return `<path${i===2&&options?.boundaries.includes(lastLevel as BoundaryLevel)?` data-boundary="${lastLevel}"`:''} d="${path(g)}" fill="${p.fill}" fill-rule="evenodd" stroke="${i===2&&options&&!options.boundaries.includes(lastLevel as BoundaryLevel)?'none':p.stroke}" stroke-width="${i===2?boundaryPaint(lastLevel as BoundaryLevel).width:p.width}"${p.dash?` stroke-dasharray="${p.dash.join(' ')}"`:''}/>`;
 };
 const leafGeometries=(g:any):any[]=>g.type==='GeometryCollection'?g.geometries.flatMap(leafGeometries):[g];
 for(const [i,layer] of [[2,admin],[1,territory]] as const){
  const shapes=layer.getFeatures().flatMap(f=>{if(i===1&&!contextBoundaryVisible(f.get('boundaryType')))return [];const geometry=format.writeGeometryObject(f.getGeometry()!);return (i===1?leafGeometries(geometry).filter(g=>contextGeometryVisible(g.type,f.get('renderRole'))):[geometry]).map(g=>({geometry:g,color:props.polygonColors?.[String(f.getId())],role:f.get('renderRole'),line:f.get('line')}));});
  if(i===1)shapes.sort((a,b)=>mapPaintPriority(a.geometry.type,a.role)-mapPaintPriority(b.geometry.type,b.role));
  for(const shape of shapes)s+=paintedPath(shape.geometry,i,shape.color,shape.role,shape.line);
 }
 const exportLevels=boundaryLevels.filter(level=>level!==lastLevel&&(options?options.boundaries.includes(level):props.extraBoundaries?.includes(level)));
 for(const level of [...exportLevels].reverse()){const p=boundaryPaint(level);for(const feature of boundaryFeatures.get(level)??[])s+=`<path data-boundary="${level}" d="${path(format.writeGeometryObject(feature.getGeometry()!))}" fill="none" stroke="${level===lastLevel?props.polygonColors?.[String(feature.getId())]??'#9aa9aa':p.color}" stroke-width="${p.width}"/>`;}
 for(const feature of sketch.getFeatures())s+=paintedPath(format.writeGeometryObject(feature.getGeometry()!),3);
 for(const p of props.points){const c=xy(fromLonLat([p.lon,p.lat]));s+=`<circle cx="${c[0]}" cy="${c[1]}" r="${p.multi?5.5:4}" fill="${p.unclassified?'#fff':p.color}" stroke="${p.selected?'#172c40':p.incomplete?'#526c78':'#fff'}" stroke-width="${p.selected||p.incomplete?1.5:1}"${p.incomplete&&!p.unclassified?' stroke-dasharray="2 2"':''}/>`;if(options?options.names==='all'||options.names==='selected'&&(p.chosen??p.selected):props.labels)s+=`<text data-place-name="${escape(p.id)}" x="${c[0]}" y="${c[1]-12}" text-anchor="middle" font-family="Microsoft YaHei,sans-serif" font-size="12" paint-order="stroke" stroke="#fff" stroke-width="3" stroke-linejoin="round" fill="#203445">${escape(p.name)}</text>`;}
 s+='</g>';s+=`<g font-family="PTB-Map-IPA,Microsoft YaHei,sans-serif" fill="#203445">`;
 const scale=target.value?.querySelector<HTMLElement>('.ol-scale-line-inner');if(scale&&options?.scale!==false){const width=scale.offsetWidth*factor;s+=`<g data-map-scale="metric"><rect x="12" y="${h-46}" width="${width+16}" height="38" rx="3" fill="#fff"/><path d="M20 ${h-23}v10h${width}v-10" fill="none" stroke="#203445"/><text x="${20+width/2}" y="${h-26}" text-anchor="middle" font-size="12">${escape(scale.textContent??'')}</text></g>`;}
 titleLines.forEach((line,i)=>{s+=`<text x="16" y="${h+30+i*28}" font-size="22">${escape(line)}</text>`;});
 legend.forEach((v,i)=>{const x=16+(i%cols)*(w/cols),y=legendTop+rowHeights.slice(0,Math.floor(i/cols)).reduce((a,b)=>a+b,0);s+=`<circle cx="${x+4}" cy="${y-4}" r="4" fill="${v.color}"/>`;legendLines[i].forEach((line,j)=>{s+=`<text x="${x+15}" y="${y+j*20}" font-size="14">${escape(line)}</text>`;});});
 sourceLines.forEach((line,i)=>{s+=`<text x="16" y="${h+footer-12-(sourceLines.length-i-1)*18}" font-size="12">${escape(line)}</text>`;});return s+'</g></svg>';
}
defineExpose({exportSvg,reload,prepareExport});
</script>
<template><div class="geo-map"><div ref="target" class="map-target" tabindex="0" aria-label="语言分布地图，滚轮缩放，Shift 拖动框选"/><span v-if="loading" class="map-loading">载入边界…</span><span v-if="hover" class="map-hover" data-background-surface>{{hover}}</span><span class="map-instructions">滚轮缩放 · Shift 拖动框选</span>
 <div v-if="attributions?.length" class="map-attributions" :class="{open:attributionOpen}" aria-label="底图来源" @keydown="attributionKeydown">
  <a v-if="osmAttribution" class="map-osm-credit" :href="osmAttribution.url" target="_blank" rel="noopener noreferrer" :title="mapAttributionText([osmAttribution])">© OpenStreetMap</a>
  <button ref="attributionButton" type="button" class="map-sources-toggle" aria-label="地图来源" :aria-expanded="attributionOpen" :aria-controls="attributionId" @click="toggleAttribution"><span aria-hidden="true">{{attributionOpen?'▾':'▸'}}</span> 地图来源</button>
  <Teleport :to="attributionHost"><ul :id="attributionId" class="map-attribution-details" data-background-surface :hidden="!attributionOpen" :style="attributionStyle" aria-label="地图来源与许可" @keydown="attributionKeydown"><li v-for="item in attributions" :key="item.id"><a :href="item.url" target="_blank" rel="noopener noreferrer">{{item.label}}</a><span v-if="item.license" class="map-source-license">{{item.license}}</span><span v-if="item.details" class="map-source-detail">{{item.details}}</span></li></ul></Teleport>
 </div>
</div></template>
<style scoped>
.map-attributions{position:absolute;right:8px;bottom:37px;display:flex;align-items:center;justify-content:flex-end;flex-wrap:wrap;gap:3px 6px;max-width:calc(100% - 16px);box-sizing:border-box;padding:3px 5px;border:1px solid #bcc7c9;border-radius:4px;background:#fffffff5;font-size:11px;line-height:1.4;color:#203445}.map-attributions a,.map-attribution-details a{color:inherit;text-decoration:underline;text-underline-offset:2px}.map-osm-credit{white-space:nowrap}.map-attributions .map-sources-toggle{display:inline-flex;align-items:center;gap:3px;min-height:24px;min-width:0;padding:2px 5px;border:1px solid #bcc7c9;border-radius:3px;background:#fff;color:#203445;font-size:11px;line-height:1.2;white-space:nowrap}.map-attributions :is(a,button):focus-visible,.map-attribution-details a:focus-visible{outline:2px solid #235c91;outline-offset:2px}.map-attribution-details{position:fixed;z-index:80;box-sizing:border-box;overflow:auto;overscroll-behavior:contain;margin:0;padding:7px 9px;border:1px solid #bcc7c9;border-radius:4px;background:var(--panel,#fff);color:var(--text,#203445);font-size:11px;line-height:1.4;box-shadow:0 2px 8px #20344525;list-style:none;text-align:left;overflow-wrap:anywhere}.map-attribution-details[hidden]{display:none}.map-attribution-details li+li{margin-top:7px;padding-top:6px;border-top:1px solid var(--border,#dbe2e4)}.map-source-license,.map-source-detail{display:block;margin-top:2px;color:var(--muted,#405463)}
.geo-map{position:relative;width:100%;height:100%;min-height:220px;background:#e5eff2;color:#203445}.map-target{width:100%;height:100%}.map-loading,.map-hover,.map-instructions{position:absolute;background:#fffffff0;border:1px solid #bcc7c9;border-radius:4px;padding:4px 8px;font-size:11px;pointer-events:none}.map-loading{left:50px;top:10px}.map-hover{left:50%;top:10px;transform:translateX(-50%)}.map-instructions{right:8px;bottom:8px}.geo-map :deep(.ol-control button){min-height:24px;min-width:24px;padding:0;color:#203445;background:#fff}.geo-map :deep(.ol-zoom){top:8px;left:8px}.geo-map :deep(.ol-scale-line){background:#ffffffdc}.geo-map :deep(.ol-scale-line-inner){color:#203445;border-color:#203445}
.map-hover{background:var(--panel,#fff);color:var(--text,#203445);border-color:var(--border,#bcc7c9)}
.geo-map :deep(.ol-scale-line){background:#fff;padding:5px 7px;border:1px solid #bcc7c9;border-radius:4px}
</style>
