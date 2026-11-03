import type {LocaleStore} from '../i18n/index.ts';
import type {AccountPreferences} from './account-preferences.ts';

/** Capture the account instance, never a mutable global account target. */
export function accountLocaleStore(session:AccountPreferences):LocaleStore {
 return {
  read:<T>(key:string,fallback:T):T=>(session.read('APP')[key] as T|undefined)??fallback,
  write:(key,value)=>session.write('APP',{[key]:value}),
 };
}
