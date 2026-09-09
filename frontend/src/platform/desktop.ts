import { browser } from './browser.ts';
import type { HostCapabilities } from './types.ts';
// Qt WebEngine implements the file input picker; no filesystem path crosses this boundary.
export const desktop:HostCapabilities={...browser,kind:'desktop'};
export function platform():HostCapabilities {return location.protocol==='ptbapp:'?desktop:browser;}
