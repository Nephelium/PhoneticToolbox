// source_id: SRC-PRAAT. Binding only; scientific algorithms remain Praat 6.1.38.
#include "Sound.h"
#include "Sound_and_Spectrogram.h"
#include "Sound_to_Pitch.h"
#include "praat.h"
#include <vector>
#include <string>
#include <emscripten/emscripten.h>

static std::vector<double> values;
static double grid[8];
static std::string error;
static void initialize() {
    static bool initialized = false;
    if (!initialized) { praatlib_init(); initialized = true; }
}

extern "C" {
EMSCRIPTEN_KEEPALIVE const double *ptb_values() { return values.data(); }
EMSCRIPTEN_KEEPALIVE const double *ptb_grid() { return grid; }
EMSCRIPTEN_KEEPALIVE const char *ptb_error() { return error.c_str(); }
EMSCRIPTEN_KEEPALIVE int ptb_spectrogram(const double *samples, int frames,
        double rate, double start, double window, double fmax, double dt, double df) {
    error.clear(); values.clear();
    if (!samples || frames < 1 || frames > 32000000 || rate <= 0 || rate > 192000)
        return -1;
    try {
        initialize();
        auto sound = Sound_create(1, start, start + frames / rate, frames, 1.0 / rate, start + 0.5 / rate);
        for (int i = 0; i < frames; ++i) sound->z[1][i + 1] = samples[i];
        auto spec = Sound_to_Spectrogram(sound.get(), window, fmax, dt, df,
            kSound_to_Spectrogram_windowShape::GAUSSIAN, 8.0, 8.0);
        if (spec->nx * spec->ny > 1000000) return -2;
        grid[0] = spec->nx; grid[1] = spec->ny;
        grid[2] = spec->x1; grid[3] = spec->dx;
        grid[4] = spec->y1; grid[5] = spec->dy;
        values.reserve(spec->nx * spec->ny);
        for (integer y = 1; y <= spec->ny; ++y)
            for (integer x = 1; x <= spec->nx; ++x) values.push_back(spec->z[y][x]);
        return 0;
    } catch (const MelderError &) {
        error = Melder_peek32to8(Melder_getError()); Melder_clearError(); return -3;
    }
}
EMSCRIPTEN_KEEPALIVE int ptb_pitch(const double *samples, int frames, double rate,
        double start, double dt, double low, double high) {
    error.clear(); values.clear();
    if (!samples || frames < 1 || frames > 32000000 || rate <= 0 || rate > 192000) return -1;
    try {
        initialize();
        auto sound = Sound_create(1, start, start + frames / rate, frames, 1.0 / rate, start + 0.5 / rate);
        for (int i=0; i<frames; ++i) sound->z[1][i+1] = samples[i];
        auto pitch = Sound_to_Pitch_ac(sound.get(), dt, low, 3.0, 15, false, .03, .45, .01, .35, .14, high);
        grid[0]=pitch->nx; grid[1]=1; grid[2]=pitch->x1; grid[3]=pitch->dx;
        for (integer i=1; i<=pitch->nx; ++i) values.push_back(pitch->frames[i].candidates[1].frequency);
        return 0;
    } catch (const MelderError &) {
        error=Melder_peek32to8(Melder_getError()); Melder_clearError(); return -3;
    }
}
}
int main() { return 0; }
