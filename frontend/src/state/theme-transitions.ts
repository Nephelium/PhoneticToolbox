/** Theme colours change together. Hover effects resume after the next paint. */
let installed=false;
export function installThemeTransitions(){
 if(installed)return;installed=true;const root=document.documentElement;let first=0,second=0;
 new MutationObserver(()=>{
  cancelAnimationFrame(first);cancelAnimationFrame(second);root.dataset.themeSwitching='';
  // Flush the new palette with transitions disabled, then restore hover effects.
  getComputedStyle(root).color;
  first=requestAnimationFrame(()=>{second=requestAnimationFrame(()=>{delete root.dataset.themeSwitching;});});
 }).observe(root,{attributes:true,attributeFilter:['data-theme','data-palette','data-custom-palette']});
}
