/** Playback-only peak gain. Never edit the source PCM or exported assets. */
export const AUDITION_PEAK=10**(-1/20),MAX_AUDITION_GAIN=32;
export function peakGain(peak:number,enabled=true){return !enabled||!Number.isFinite(peak)||peak<=1e-5?1:Math.min(MAX_AUDITION_GAIN,AUDITION_PEAK/peak);}
export function samplePeak(channels:readonly ArrayLike<number>[]){let peak=0;for(const channel of channels)for(let i=0;i<channel.length;i++){const value=channel[i]!;if(!Number.isFinite(value))return Infinity;peak=Math.max(peak,Math.abs(value));}return peak;}
export function bufferPeak(buffer:AudioBuffer){return samplePeak(Array.from({length:buffer.numberOfChannels},(_,i)=>buffer.getChannelData(i)));}
