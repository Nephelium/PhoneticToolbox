import {dark} from './theme.js';
import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {planeSegments,stitch,oralMesh,connectedAirway,nasalOutlet,sectionPaths,airwayGuide,displayLips} from './geometry.mjs';
import {registerNose,velumModel} from './anatomy.mjs';
const NS='http://www.w3.org/2000/svg';
const el=(name,attrs={})=>{const e=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e;};
const colors={upper_cover:'#c6a28e',lower_cover:'#c6a28e',upper_teeth:'#f5edda',lower_teeth:'#f5edda',upper_lip:'#bf757b',lower_lip:'#bf757b',tongue:'#c58b8c',uvula:'#b07783',epiglottis:'#b98b80'};
const path=(pts,close=false)=>pts?.length?'M'+pts.map(p=>p[0]+','+(-p[1])).join('L')+(close?'Z':''):'';
function geometry(data){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(data.vertices,3));g.setIndex(data.triangles);g.computeVertexNormals();g.computeBoundingSphere();return g;}
export class VocalTractViewer{
  constructor(viewport,onDrag,{onStart=()=>{},onEnd=()=>{},onSelect=()=>{}}={}){
    Object.assign(this,{el:viewport,onDrag,onStart,onEnd,onSelect,sagittal:true,focused:true,labels:true,showHead:true,showNose:true,showTeeth:false,mode:'organs',selected:'tongue',zoom:1,pan:[0,0],state:null});
    this.svg=el('svg',{class:'sagittal-canvas','aria-label':'正中矢状截面'});viewport.append(this.svg);
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.localClippingEnabled=true;this.renderer.setClearColor(dark()?0x192d39:0xf5f3eb,1);this.renderer.toneMapping=THREE.ACESFilmicToneMapping;viewport.append(this.renderer.domElement);
    this.scene=new THREE.Scene();this.camera=new THREE.PerspectiveCamera(34,1,.1,200);this.orbit=new OrbitControls(this.camera,this.renderer.domElement);this.orbit.enableDamping=true;this.orbit.minDistance=12;this.orbit.maxDistance=90;
    this.scene.add(new THREE.HemisphereLight(0xfff9ed,0x627c74,2));for(const [power,pos] of [[2,[10,20,25]],[1.5,[-15,8,-15]]]){const light=new THREE.DirectionalLight(0xfff9eb,power);light.position.set(...pos);this.scene.add(light);}
    this.halfPlane=new THREE.Plane(new THREE.Vector3(0,0,-1),0);this.meshes=new Map();this.controls=new Map();this.raycaster=new THREE.Raycaster();this.dragPlane=new THREE.Plane(new THREE.Vector3(0,0,1),0);
    this.controlLabels=document.createElement('div');this.controlLabels.className='control-labels';viewport.append(this.controlLabels);
    for(const name of ['tongue','blade','tip','root','hyoid','upper_lip','lower_lip','velum','side1','side2','side3']){
      const mesh=new THREE.Mesh(new THREE.SphereGeometry(1,14,10),new THREE.MeshBasicMaterial({color:0x237768,depthTest:false}));mesh.userData.handle=name;mesh.renderOrder=12;this.scene.add(mesh);
      const label=document.createElement('span');this.controlLabels.append(label);this.controls.set(name,{mesh,label});
    }
    this.valve=new THREE.Mesh(new THREE.CircleGeometry(1,40),new THREE.MeshStandardMaterial({color:0xc1868b,side:THREE.DoubleSide,roughness:.7}));this.valve.rotation.x=Math.PI/2;this.valve.userData.organ='velum';this.scene.add(this.valve);
    this.svg.addEventListener('pointerdown',e=>{
      if(!this.state)return;
      const h=e.target.closest('[data-handle]');
      if(h&&!this.playing&&e.button===0&&!e.shiftKey){this.onStart();this.dragging={name:h.dataset.handle,start:this.svgPoint(e),params:this.dragBaseline(h.dataset.handle)};}
      else {const o=e.target.closest('[data-organ]');if(o&&!this.playing&&e.button===0)this.select(o.dataset.organ);
        this.panning={start:[e.clientX,e.clientY],pan:[...this.pan],scale:this.svg.viewBox.baseVal.height/this.el.clientHeight};this.svg.classList.add('panning');}
      this.svg.setPointerCapture(e.pointerId);e.preventDefault();
    });
    this.svg.addEventListener('pointermove',e=>{
      if(this.panning){const d=this.panning;this.pan=[d.pan[0]-(e.clientX-d.start[0])*d.scale,d.pan[1]+(e.clientY-d.start[1])*d.scale];this.draw2D();}
      else if(this.dragging){const p=this.svgPoint(e),d=this.dragging;this.onDrag(d.name,p[0]-d.start[0],p[1]-d.start[1],d.params);}
    });
    for(const event of ['pointerup','pointercancel','lostpointercapture'])this.svg.addEventListener(event,()=>{this.panning=null;this.svg.classList.remove('panning');if(this.dragging){this.dragging=null;this.onEnd();}});
    this.svg.addEventListener('wheel',e=>{e.preventDefault();const before=this.svgPoint(e);this.zoom=Math.max(.65,Math.min(2.8,this.zoom*(e.deltaY<0?1.08:.925)));this.draw2D();const after=this.svgPoint(e);this.pan=[this.pan[0]+before[0]-after[0],this.pan[1]+before[1]-after[1]];this.draw2D();},{passive:false});
    const ray=e=>{const r=this.renderer.domElement.getBoundingClientRect();this.raycaster.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),this.camera);};
    this.renderer.domElement.addEventListener('pointerdown',e=>{if(!this.state||this.playing)return;ray(e);
      // Pick in screen space so targets remain usable at every zoom level.
      const r=this.renderer.domElement.getBoundingClientRect();let handle=null,best=15;
      for(const [name,{mesh}] of this.controls){if(!mesh.visible)continue;const p=mesh.position.clone().project(this.camera),d=Math.hypot((p.x+1)*r.width/2+r.left-e.clientX,(1-p.y)*r.height/2+r.top-e.clientY);if(d<best){best=d;handle=name;}}
      if(handle){const m=this.controls.get(handle).mesh;this.dragPlane.constant=-m.position.z;const p=this.raycaster.ray.intersectPlane(this.dragPlane,new THREE.Vector3());if(!p)return;this.onStart();this.dragging3D={name:handle,start:p,params:this.dragBaseline(handle)};this.orbit.enabled=false;this.renderer.domElement.setPointerCapture(e.pointerId);e.stopImmediatePropagation();}
      else{const hit=this.raycaster.intersectObjects([...this.meshes.values(),this.softPalate,this.palateSection,this.tongueSection].filter(m=>m?.visible))[0];if(hit?.object.userData.organ)this.select(hit.object.userData.organ);}
    },true);
    this.renderer.domElement.addEventListener('pointermove',e=>{if(!this.dragging3D)return;ray(e);const p=this.raycaster.ray.intersectPlane(this.dragPlane,new THREE.Vector3());if(p){const d=this.dragging3D;this.onDrag(d.name,p.x-d.start.x,p.y-d.start.y,d.params);}});
    for(const event of ['pointerup','pointercancel','lostpointercapture'])this.renderer.domElement.addEventListener(event,()=>{if(this.dragging3D){this.dragging3D=null;this.orbit.enabled=true;this.onEnd();}});
    this.needsRender=true;this.orbit.addEventListener('change',()=>{this.needsRender=true;});this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(viewport);this.reset();this.animate();
  }
  async load(){
    const [head,nose]=await Promise.all(['head','nasal'].map(async n=>{const r=await fetch('assets/'+n+'.json');if(!r.ok)throw Error('无法载入模型 '+n);return r.json();}));
    for(let i=0;i<head.vertices.length;i+=3){const [x,y,z]=head.vertices.slice(i,i+3);head.vertices[i]=(z-.127)*100+6.5;head.vertices[i+1]=(y-1.85)*85-1;head.vertices[i+2]=x*80;}
    this.headContour=stitch(planeSegments(head.vertices,head.triangles))[0];this.head=new THREE.Mesh(geometry(head),new THREE.MeshStandardMaterial({color:0xc6bca6,roughness:.8,transparent:true,opacity:.17,depthWrite:false,side:THREE.DoubleSide,clippingPlanes:[this.halfPlane]}));this.head.renderOrder=-1;this.scene.add(this.head);
    if(!nose.coordinate_system?.startsWith('VTL reference cm'))throw Error('鼻腔坐标版本不匹配');
    const registered=registerNose(nose);Object.assign(nose,registered);this.noseData=nose;this.outlet=nasalOutlet(nose);this.nosePaths=sectionPaths(nose,.5);
    this.noseOutline=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x679b8c,transparent:true,opacity:.48}));const points=[];
    for(const z of [-.5,.5])for(const line of planeSegments(nose.vertices,nose.triangles,z))for(const p of line)points.push(p[0],p[1],z);
    this.noseOutline.geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));this.scene.add(this.noseOutline);this.resize();
  }
  select(organ){if(!['tongue','lips','velum'].includes(organ))return;this.selected=organ;this.onSelect(organ);this.draw2D();this.style3D();}
  setOptions(options={}){for(const [key,value] of Object.entries(options)){const p={head:'showHead',nose:'showNose',teeth:'showTeeth'}[key]||key;this[p]=value;}if('focused' in options)this.reset();if(('nose' in options)&&this.state){this.airData=this.showNose?connectedAirway(this.state,this.noseData,this.outlet):oralMesh(this.state);this.airPaths=sectionPaths(oralMesh(this.state));}this.draw2D();if(!this.sagittal&&this.state)this.update3D();}
  update(state){state=displayLips(state);this.state=state;this.palate=velumModel(state);this.airData=this.showNose?connectedAirway(state,this.noseData,this.outlet):oralMesh(state);this.airPaths=sectionPaths(oralMesh(this.state));this.velumPaths=sectionPaths(this.palate);this.draw2D();if(!this.sagittal)this.update3D();}
  setView(value){this.sagittal=value;if(!value&&this.state)this.update3D();this.resize();}
  reset(){this.zoom=1;this.pan=[0,0];this.orbit.target.set(1,this.focused?-.1:4,0);this.camera.position.set(...(this.focused?[12,4.3,25.5]:[25,15,40]));this.orbit.update();this.draw2D();}
  resize(){const w=this.el.clientWidth,h=this.el.clientHeight;if(!w||!h)return;if(this.width!==w||this.height!==h){this.width=w;this.height=h;this.renderer.setSize(w,h,false);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.needsRender=true;}this.svg.style.display=this.sagittal?'block':'none';this.renderer.domElement.style.display=this.sagittal?'none':'block';this.controlLabels.hidden=this.sagittal;this.draw2D();}
  svgPoint(e){const p=this.svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const q=p.matrixTransform(this.svg.getScreenCTM().inverse());return [q.x,-q.y];}
  dragBaseline(name){const p=[...this.state.params];const keys={blade:['TBX','TBY'],root:['TRX','TRY']}[name]||[];for(const key of keys){const i=this.metadata.parameters.findIndex(m=>m.name===key);p[i]=this.state.limited[i];}return p;}
  handles(){
    const s=this.state,idx=n=>this.metadata.parameters.findIndex(p=>p.name===n),xy=(x,y)=>[s.limited[idx(x)],s.limited[idx(y)],.06];
    const items=[{name:'tongue',label:'舌背',p:xy('TCX','TCY'),key:'TCX',organ:'tongue'},{name:'blade',label:'舌叶',p:xy('TBX','TBY'),key:'TBX',organ:'tongue'},{name:'root',label:'舌根',p:xy('TRX','TRY'),key:'TRX',organ:'tongue'},{name:'hyoid',label:'舌骨',p:[...s.contours.lower_cover[4],.06],key:'HY',organ:'tongue'},{name:'tip',label:'舌尖',p:xy('TTX','TTY'),key:'TTX',organ:'tongue'},
      ...['upper','lower'].map(n=>({name:n+'_lip',label:n==='upper'?'上唇':'下唇',p:[...s.lipHandles[n+'_lip'],.06],key:'LP',organ:'lips'})),{name:'velum',label:'软腭 / 腭咽口',p:[...s.contours.uvula.at(-1),.06],key:'VO',organ:'velum'}];
    const tongue=s.meshes.find(m=>m.name==='tongue');
    for(let k=1;k<=3;k++){
      // Regional anchors on the native tongue surface, posterior to anterior.
      // VTL has 33 dynamic ribs followed by four static underside ribs.
      const rib=Math.round([.34,.66,.94][k-1]*32);
      const p=tongue.vertices.slice((rib*tongue.points+tongue.points-1)*3,(rib*tongue.points+tongue.points-1)*3+3);
      items.push({name:'side'+k,label:['后侧缘','中侧缘','前侧缘'][k-1],p,key:'TS'+k,organ:'tongue',side:true});
    }return items;
  }
  add(parent,pts,attrs={},close=false){const node=el('path',{d:path(pts,close),...attrs});parent.append(node);return node;}
  draw2D(){
    if(!this.state||!this.sagittal)return;const s=this.state,c=s.contours,px=(this.focused?17.3:26)/this.zoom/this.el.clientHeight,h=px*this.el.clientHeight,w=px*this.el.clientWidth,cx=(this.focused?1.4:-1.8)+this.pan[0],cy=(this.focused?-.1:4)+this.pan[1];
    const focus=this.svg.contains(document.activeElement)?document.activeElement.dataset.handle:null;
    this.svg.setAttribute('viewBox',[cx-w/2,-cy-h/2,w,h].join(' '));this.svg.replaceChildren();
    const defs=el('defs');defs.innerHTML='<linearGradient id="tongueFill" x2="0.2" y2="1"><stop stop-color="#e0adab"/><stop offset="1" stop-color="#b9797c"/></linearGradient>';this.svg.append(defs);
    const layer=el('g',{'stroke-linejoin':'round','stroke-linecap':'round'});this.svg.append(layer);
    if(this.showHead)this.add(layer,this.headContour,{fill:'#eadfcd',stroke:'#d4c8b5','stroke-width':.03,opacity:.43},true);
    const air=this.mode==='airway',overlay=this.mode==='overlay',airFill=air?'#a8d1c3':overlay?'#dce9df':'#fffdf6',tissueOpacity=air?.15:1;
    if(this.showNose&&this.nosePaths)layer.append(el('path',{d:this.nosePaths.map(p=>path(p,false)).join(' '),fill:air||overlay?'#a8d1c3':'#eff2e8','fill-opacity':air?.9:.4,'fill-rule':'evenodd',stroke:'#72a394','stroke-width':.035,'stroke-dasharray':'.09 .06','data-role':'nasal-reference','data-registration':'nasal/3'}));
    if(this.showNose&&this.airData.connector){const lines=sectionPaths(this.airData.connector);layer.append(el('path',{d:lines.map(p=>path(p,true)).join(' '),fill:air||overlay?'#a8d1c3':'#eff2e8',stroke:'#91ac9d','stroke-width':.025,'fill-rule':'evenodd','data-role':'nasopharyngeal-connection'}));}
    layer.append(el('path',{d:this.airPaths.map(p=>path(p,true)).join(' '),fill:airFill,stroke:'#91ac9d','stroke-width':.035,'fill-rule':'evenodd','data-role':'midsagittal-airway'}));
    for(const pts of [c.upper_cover.slice(0,8),c.upper_cover.slice(13),c.lower_cover])this.add(layer,pts,{fill:'none',stroke:'#c9a593','stroke-width':.14,opacity:tissueOpacity});
    this.add(layer,s.tongueOutline,{fill:air?'#e9ded0':'url(#tongueFill)',stroke:this.selected==='tongue'?'#93595f':'#b98587','stroke-width':this.selected==='tongue'?.065:.035,'data-organ':'tongue','data-role':'tongue-section',cursor:'pointer'},true);
    // Hidden teeth still constrain the airway and native acoustic calculations.
    if(this.showTeeth)for(const n of ['upper_teeth','lower_teeth'])this.add(layer,c[n],{fill:'#fff9e9',stroke:'#b6a486','stroke-width':.045,'data-role':'incisor-section'},true);
    for(const n of ['upper_lip','lower_lip']){this.add(layer,c[n],{fill:colors[n],stroke:this.selected==='lips'?'#93595f':'#a96a70','stroke-width':.045,opacity:tissueOpacity,'data-organ':'lips',cursor:'pointer'},true);}
    layer.append(el('path',{d:this.velumPaths.map(p=>path(p,true)).join(' '),fill:'#c1848e',stroke:this.selected==='velum'?'#855566':'#a96f7c','stroke-width':.035,'fill-rule':'evenodd','data-organ':'velum','data-role':s.nasal.port_area>0?'open-nasal-port':'closed-nasal-port',cursor:'pointer',opacity:air?.65:1}));
    this.add(layer,c.epiglottis,{fill:'none',stroke:'#b77f76','stroke-width':.13});
    const guides=airwayGuide(s);let segment=[];
    const drawGuide=()=>{if(segment.length>1)this.add(layer,segment,{fill:'none',stroke:'#559386','stroke-width':.028,'stroke-dasharray':'.14 .12','data-role':'airway-guide'});segment=[];};
    for(const q of guides){if(q)segment.push(q.point);else drawGuide();}drawGuide();
    const q=guides[s.section];if(q)this.add(layer,[q.lower,q.upper],{stroke:'#327e70','stroke-width':.045,'stroke-dasharray':'.12 .1','data-role':'section-guide'});
    if(this.labels)for(const [label,p,offset] of [['硬腭',c.upper_cover[17],[.1,1]],['软腭 / 小舌',this.palate.tip,[-2.6,1.4]],['会厌',c.epiglottis.at(-1),[-2.2,-.5]],...(this.showTeeth?[['门齿截面',c.upper_teeth[2],[2.8,1.4]]]:[]),...(this.showNose?[['鼻腔参考 · 偏离中线 5 mm',[2,4.7],[1,1.4]]]:[])])this.annotation(layer,label,p,offset,px);
    const offsets={tongue:[.25,.8],blade:[-.2,-.65],root:[-1.5,.1],hyoid:[-1.2,.5],tip:[.4,.7],upper_lip:[.6,-.35],lower_lip:[.6,.75],velum:[-2,.9]};
    for(const item of this.handles().filter(p=>!p.side&&p.organ===this.selected)){
      const {name,label,p,key}=item,index=this.metadata.parameters.findIndex(m=>m.name===key),param=this.metadata.parameters[index];
      const g=el('g',{'data-handle':name,role:'slider','aria-label':'拖动'+label,'aria-valuemin':param.min,'aria-valuemax':param.max,'aria-valuenow':s.params[index],tabindex:0,class:'organ-handle',transform:'translate('+p[0]+' '+(-p[1])+')'});
      g.append(el('circle',{r:px*16,fill:'transparent'}),el('circle',{r:px*6,fill:'#fffdf5',stroke:'#327d6d','stroke-width':px*1.6}),el('circle',{r:px*2.5,fill:'#327d6d'}));
      if(this.labels){const [x,y]=offsets[name],text=el('text',{x,y,'font-size':px*11,fill:'#356f61','paint-order':'stroke',stroke:'#fffdf6','stroke-width':px*2});text.textContent=label;g.append(text);}layer.append(g);
      g.addEventListener('keydown',e=>{if(this.playing||!e.key.startsWith('Arrow'))return;e.preventDefault();this.onStart();const d=e.shiftKey?.2:.04;this.onDrag(name,e.key==='ArrowLeft'?-d:e.key==='ArrowRight'?d:0,e.key==='ArrowUp'?d:e.key==='ArrowDown'?-d:0,[...this.state.params]);this.onEnd();});
      if(name===focus)g.focus({preventScroll:true});
    }
    if(dark()){
      const palette={'#eadfcd':'#82938b','#d4c8b5':'#6c827b','#fffdf6':'#1b303b','#fffdf5':'#203b45','#f8f5ec':'#1b303b','#eff2e8':'#2c4b4e','#a8d1c3':'#426f6a','#dce9df':'#345452','#91ac9d':'#669c93','#e9ded0':'#40545a','#c9a593':'#b59889','#93595f':'#e8a0a8','#b98587':'#cf8994','#356f61':'#a2ddd1','#6a7969':'#a6c8be','#99a48f':'#68948d','#559386':'#71bcae','#327e70':'#86dbca','#327d6d':'#80d4c4','#72a394':'#74a69d','#fff9e9':'#c7cec5'};
      for(const node of this.svg.querySelectorAll('*'))for(const attr of ['fill','stroke']){const v=node.getAttribute(attr);if(palette[v])node.setAttribute(attr,palette[v]);}
      this.svg.querySelector('stop')?.setAttribute('stop-color','#b57d89');this.svg.querySelector('stop[offset="1"]')?.setAttribute('stop-color','#875866');
    }
    this.orientation();
  }
  annotation(parent,label,p,offset,px){const end=[p[0]+offset[0],p[1]+offset[1]];this.add(parent,[p,end],{fill:'none',stroke:'#99a48f','stroke-width':.025});const text=el('text',{x:end[0],y:-end[1]-.12,'font-size':px*10.5,fill:'#6a7969','text-anchor':offset[0]<0?'end':'start','paint-order':'stroke',stroke:'#f8f5ec','stroke-width':px*2});text.textContent=label;parent.append(text);}
  update3D(){
    const s=this.state;for(const data of s.meshes){let mesh=this.meshes.get(data.name);if(!mesh){mesh=new THREE.Mesh(geometry(data),new THREE.MeshStandardMaterial({color:colors[data.name],roughness:.7,side:THREE.DoubleSide}));mesh.userData.organ=data.name==='tongue'?'tongue':data.name.includes('lip')?'lips':data.name==='uvula'?'velum':null;this.meshes.set(data.name,mesh);this.scene.add(mesh);}else{mesh.geometry.attributes.position.array.set(data.vertices);mesh.geometry.attributes.position.needsUpdate=true;mesh.geometry.computeVertexNormals();mesh.geometry.computeBoundingSphere();}
    }
    // The native upper cover includes a mathematical back-wall bridge; remove
    // its velum cells and display the shared soft-tissue reconstruction instead.
    const cover=s.meshes.find(m=>m.name==='upper_cover');const kept=[];for(let i=0;i<cover.triangles.length;i+=3){const tri=cover.triangles.slice(i,i+3),ribs=tri.map(n=>Math.floor(n/cover.points));if(Math.max(...ribs)<=7||Math.min(...ribs)>=13)kept.push(...tri);}this.meshes.get('upper_cover').geometry.setIndex(kept);
    if(this.softPalate){this.softPalate.geometry.dispose();this.softPalate.geometry=geometry(this.palate);}else{this.softPalate=new THREE.Mesh(geometry(this.palate),new THREE.MeshStandardMaterial({color:0xc1848e,side:THREE.DoubleSide,roughness:.8}));this.softPalate.userData.organ='velum';this.scene.add(this.softPalate);}
    const palateCap=new THREE.ShapeGeometry(new THREE.Shape(this.palate.profile.map(p=>new THREE.Vector2(...p))));if(this.palateSection){this.palateSection.geometry.dispose();this.palateSection.geometry=palateCap;}else{this.palateSection=new THREE.Mesh(palateCap,new THREE.MeshStandardMaterial({color:0xc1848e,side:THREE.DoubleSide,roughness:.8}));this.palateSection.position.z=.01;this.palateSection.userData.organ='velum';this.scene.add(this.palateSection);}
    // A midline display cap makes the cut tongue legible as tissue. Its lower
    // boundary is the native mouth floor; it is not a measured tissue volume.
    const outline=s.tongueOutline;
    const capGeometry=new THREE.ShapeGeometry(new THREE.Shape(outline.map(p=>new THREE.Vector2(...p))));
    if(!this.tongueSection){this.tongueSection=new THREE.Mesh(capGeometry,new THREE.MeshStandardMaterial({color:colors.tongue,side:THREE.DoubleSide,roughness:.8}));this.tongueSection.position.z=.015;this.tongueSection.userData.organ='tongue';this.scene.add(this.tongueSection);}else{this.tongueSection.geometry.dispose();this.tongueSection.geometry=capGeometry;}
    if(this.airway){this.scene.remove(this.airway);this.airway.geometry.dispose();for(const m of this.airway.material)m.dispose();}
    const data=this.airData,g=geometry(data);this.connection=data.junction;
    g.clearGroups();if(this.showNose){g.addGroup(0,data.oralIndexCount,0);g.addGroup(data.oralIndexCount,this.noseData.triangles.length,1);g.addGroup(data.oralIndexCount+this.noseData.triangles.length,Infinity,2);}else g.addGroup(0,data.triangles.length,0);
    this.airway=new THREE.Mesh(g,[0,1,2].map(()=>new THREE.MeshStandardMaterial({color:0x6faaa0,roughness:.55,side:THREE.DoubleSide,transparent:true,depthWrite:false})));this.scene.add(this.airway);
    this.valve.visible=false;
    this.el.dataset.junctionOpen=this.connection?.open?'true':'false';this.el.dataset.junctionArea=s.nasal.port_area;
    this.style3D();
  }
  style3D(){
    this.needsRender=true;this.renderer.setClearColor(dark()?0x192d39:0xf5f3eb,1);
    if(!this.state)return;const air=this.mode==='airway',overlay=this.mode==='overlay';
    for(const [name,mesh] of this.meshes){const cover=name.includes('cover');mesh.visible=name==='uvula'?false:name.includes('teeth')?this.showTeeth:!air;
      mesh.material.clippingPlanes=cover||name==='tongue'?[this.halfPlane]:[];mesh.material.transparent=air||overlay||cover;mesh.material.opacity=air?.12:cover?.2:overlay?.64:1;mesh.material.depthWrite=!mesh.material.transparent;
      mesh.material.emissive.set(mesh.userData.organ===this.selected?0x22130e:0);mesh.material.emissiveIntensity=.15;
    }
    if(this.softPalate){for(const mesh of [this.softPalate,this.palateSection]){mesh.visible=true;mesh.material.clippingPlanes=mesh===this.softPalate?[this.halfPlane]:[];mesh.material.transparent=overlay;mesh.material.opacity=overlay?.72:1;}}
    if(this.airway){const opacities=air?[1,1,1]:overlay?[.25,.5,.65]:[.025,.10,0];this.airway.material.forEach((m,i)=>{m.opacity=opacities[i];m.transparent=!air;m.depthWrite=air;});}
    if(this.tongueSection){this.tongueSection.visible=!air;this.tongueSection.material.transparent=overlay;this.tongueSection.material.opacity=overlay?.64:1;this.tongueSection.material.depthWrite=!overlay;}
    if(this.head){this.head.visible=this.showHead;this.head.material.color.set(dark()?0x91a3a2:0xc6bca6);}if(this.noseOutline)this.noseOutline.visible=this.showNose&&!air;
    const handles=this.handles();for(const [name,control] of this.controls){const h=handles.find(p=>p.name===name);control.mesh.visible=h.organ===this.selected;control.mesh.position.set(...h.p);control.label.textContent=h.label;control.label.hidden=!control.mesh.visible||!this.labels;}
  }
  orientation(){const compass=this.el.querySelector('.orientation');if(!compass)return;if(this.sagittal){compass.innerHTML='<span>上</span><div>后 ── 前</div><span>下</span>';return;}
    const q=this.camera.quaternion.clone().invert(),axes=[['前',new THREE.Vector3(1,0,0)],['上',new THREE.Vector3(0,1,0)],['侧',new THREE.Vector3(0,0,1)]];
    compass.innerHTML='<svg width="78" height="70" viewBox="0 0 78 70">'+axes.map(([name,v])=>{v.applyQuaternion(q);return '<line x1="34" y1="36" x2="'+(34+v.x*23)+'" y2="'+(36-v.y*23)+'" stroke="#91a293"/><text x="'+(34+v.x*30)+'" y="'+(39-v.y*30)+'" text-anchor="middle" fill="#718776" font-size="10">'+name+'</text>';}).join('')+'</svg>';
  }
  renderNow(){if(!this.sagittal){const h=this.el.clientHeight,w=this.el.clientWidth;
      for(const [name,{mesh,label}] of this.controls){if(!mesh.visible)continue;const size=2*Math.tan(this.camera.fov*Math.PI/360)*this.camera.position.distanceTo(mesh.position)/h;mesh.scale.setScalar(size*5.5);const p=mesh.position.clone().project(this.camera);const offset={tongue:[-42,9],blade:[-20,-31],root:[-1.5,.1],hyoid:[-1.2,.5],tip:[10,7],side1:[-52,-25],side2:[-6,13],side3:[14,-27]}[name]||[11,-12];label.style.left=((p.x+1)*w/2+offset[0])+'px';label.style.top=((1-p.y)*h/2+offset[1])+'px';}
      this.orientation();this.renderer.render(this.scene,this.camera);
      this.needsRender=false;this.el.dataset.renderCount=String(1+(+this.el.dataset.renderCount||0));
    }}
  animate(){this.animationFrame=requestAnimationFrame(()=>this.animate());if(!this.sagittal&&!document.hidden&&!document.querySelector('dialog[open]')){this.orbit.update();if(this.needsRender)this.renderNow();}}
  dispose(){cancelAnimationFrame(this.animationFrame);this.observer.disconnect();this.orbit.dispose();this.scene.traverse(o=>{o.geometry?.dispose();for(const m of (Array.isArray(o.material)?o.material:[o.material]))m?.dispose();});this.renderer.dispose();this.el.remove();}
}
