"""Small in-memory bridge to the pinned original Praat C++ WASM build.

Only the two used Sound methods are exposed. No file I/O or approximation.
source_id: SRC-PRAAT; compatibility surface: praat-parselmouth 0.4.7.
"""
import numpy as np

__version__ = '0.4.7'
PRAAT_VERSION = '6.1.38'
module = None


class SpectralAnalysisWindowShape:
    GAUSSIAN = 'GAUSSIAN'


class Grid:
    def __init__(self, values, grid, pitch=False):
        nx, ny = int(grid[0]), int(grid[1])
        self.x1, self.dx, self.y1, self.dy = map(float, grid[2:6])
        self.values = values.reshape(ny, nx)
        self.selected_array = {'frequency': values} if pitch else {}
    def xs(self):return self.x1+np.arange(self.values.shape[1])*self.dx
    def ys(self):return self.y1+np.arange(self.values.shape[0])*self.dy


class Sound:
    def __init__(self, values, sampling_frequency=44100., start_time=0.):
        self.samples=np.asarray(values,dtype=np.float64)
        if self.samples.ndim!=1:raise ValueError('Browser Praat bridge requires one selected channel')
        self.rate=float(sampling_frequency);self.start=float(start_time)
    def _call(self, name, args, pitch=False):
        if module is None:raise RuntimeError('Praat WASM is not initialized')
        from js import Uint8Array
        pointer=module._malloc(self.samples.nbytes)
        try:
            module.HEAPU8.set(Uint8Array.new(self.samples.tobytes()),pointer)
            status=getattr(module,name)(pointer,len(self.samples),self.rate,self.start,*args)
            if status:raise ValueError(str(module.UTF8ToString(module._ptb_error())))
            grid=np.array(module.HEAPF64.subarray(module._ptb_grid()//8,module._ptb_grid()//8+8).to_py(),copy=True)
            count=int(grid[0]*grid[1]);offset=module._ptb_values()//8
            values=np.array(module.HEAPF64.subarray(offset,offset+count).to_py(),copy=True)
            return Grid(values,grid,pitch)
        finally:module._free(pointer)
    def to_spectrogram(self, *, window_length, maximum_frequency, time_step, frequency_step, window_shape):
        if window_shape!=SpectralAnalysisWindowShape.GAUSSIAN:raise ValueError('Unsupported window shape')
        return self._call('_ptb_spectrogram',(window_length,maximum_frequency,time_step,frequency_step))
    def to_pitch(self, *, time_step, pitch_floor, pitch_ceiling):
        return self._call('_ptb_pitch',(time_step,pitch_floor,pitch_ceiling),True)
