import catalog from '../generated/application-icons.json' with {type:'json'};
export const iconCatalog=catalog;
export const builtinIconIds=new Set(catalog.icons.map(item=>item.id));
