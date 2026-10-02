"""Explicit synthetic PortAudio test double; never opens a physical device."""
import threading
import time
import numpy as np

class CallbackStop(Exception):pass
class CallbackAbort(Exception):pass

class Stream:
    def __init__(self,output=False,**kwargs):
        self.options=kwargs;self.samplerate=kwargs['samplerate'];self.channels=kwargs['channels'];self.active=False;self.thread=None;self.output=output;self.frames=0;self.closed=False
    def start(self):
        self.active=True;self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        rng=np.random.default_rng(1602)
        try:
            while self.active:
                n=self.options.get('blocksize',1024);t=(np.arange(n)+self.frames)/self.samplerate
                if self.output:
                    out=np.zeros((n,1),np.float32);self.options['callback'](out,n,None,'')
                else:
                    mic=rng.normal(0,.012,n)+np.where(t>.3,.15*np.sin(2*np.pi*180*t),0)
                    cols=[mic]+[.45*np.sin(2*np.pi*(90+i*20)*t) for i in range(self.channels-1)]
                    self.options['callback'](np.column_stack(cols).astype(np.float32),n,None,'')
                self.frames+=n;time.sleep(.008)
        except (CallbackStop,CallbackAbort):pass
        finally:
            self.active=False
            if self.options.get('finished_callback'):self.options['finished_callback']()
    def stop(self):
        self.active=False
        if self.thread:self.thread.join(2)
    def abort(self):self.stop()
    def close(self):self.stop();self.closed=True

class Backend:
    CallbackStop=CallbackStop;CallbackAbort=CallbackAbort
    def query_hostapis(self):return [{'name':'M16 synthetic test only'}]
    def query_devices(self):return [{'name':'合成验证声卡（非真实硬件）','hostapi':0,'max_input_channels':2,'max_output_channels':2,'default_samplerate':48000}]
    def check_input_settings(self,**kwargs):
        if kwargs['channels']>2:raise ValueError('unsupported channels')
    def check_output_settings(self,**kwargs):pass
    def InputStream(self,**kwargs):return Stream(**kwargs)
    def OutputStream(self,**kwargs):return Stream(output=True,**kwargs)
