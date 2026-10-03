import {ref} from 'vue';
import {projects} from '../platform/browser.ts';
import {normalizeButtons,type ButtonAppearance} from '../design/buttons.ts';
export const buttonAppearance=ref<ButtonAppearance>(normalizeButtons(null));
let installed=false;
function apply(){
 document.documentElement.dataset.buttonStyle=buttonAppearance.value.mode;
 document.documentElement.dataset.buttonEffects=buttonAppearance.value.effects?'on':'off';
}
export function installButtonAppearance(){
 if(installed)return;installed=true;
 buttonAppearance.value=normalizeButtons(projects.read<unknown>('buttonAppearance',null));apply();
}
export function setButtonAppearance(value:Partial<ButtonAppearance>){
 buttonAppearance.value=normalizeButtons({...buttonAppearance.value,...value});apply();
 return projects.write('buttonAppearance',buttonAppearance.value);
}
