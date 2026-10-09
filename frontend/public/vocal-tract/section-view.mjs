// Native profiles sample a 7 cm lateral window. Keep the same metric view
// for all poses and sections, with equal horizontal and vertical centimetres.
export function sectionView(width,height){
  const top=12,bottom=26,scale=Math.max(0,Math.min(width-32,height-top-bottom)/7);
  const centerX=width/2,centerY=(height+top-bottom)/2;
  return {scale,centerX,centerY,x:index=>centerX+(index*7/96-3.5)*scale,y:value=>centerY-value*scale};
}
