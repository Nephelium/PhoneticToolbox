// Call only after committing an edit, so rejected/clamped values match the model.
export function syncNumberInput(event:Event,value:number){
 const input=event.target as HTMLInputElement,text=String(value);
 if(input.value!==text)input.value=text;
}
