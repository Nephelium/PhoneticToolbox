/** Add a centered, neutral heading to a detached export, never to live UI. */
export function withFigureTitle(source:SVGSVGElement,title:string,fontSize=12){
  const ns='http://www.w3.org/2000/svg',width=source.viewBox.baseVal.width,height=source.viewBox.baseVal.height;
  const padding=Math.max(32,fontSize*2.5),root=document.createElementNS(ns,'svg');
  root.setAttribute('xmlns',ns);root.setAttribute('viewBox',`0 0 ${width} ${height+padding}`);
  root.setAttribute('width',String(width));root.setAttribute('height',String(height+padding));
  const bg=document.createElementNS(ns,'rect');bg.setAttribute('width','100%');bg.setAttribute('height','100%');bg.setAttribute('fill','white');root.append(bg);
  const text=document.createElementNS(ns,'text');text.textContent=title;text.setAttribute('x',String(width/2));text.setAttribute('y',String(fontSize*1.6));text.setAttribute('text-anchor','middle');
  text.style.fontFamily=source.style.fontFamily;text.style.fontSize=fontSize+'px';text.style.fill='#182332';root.append(text);
  source.setAttribute('x','0');source.setAttribute('y',String(padding));source.setAttribute('width',String(width));source.setAttribute('height',String(height));
  source.style.width='';source.style.height='';source.style.maxWidth='none';root.append(source);
  return {root,width,height:height+padding};
}
