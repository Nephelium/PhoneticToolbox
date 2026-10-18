"""Reviewed opt-in source extension to the pinned GPL VTL 2.4 sources.

The unconfigured branch retains upstream jitter/rand and amplitude arithmetic.
No generated sources or bundled native binaries are edited by this module.
"""
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old)!=1:raise ValueError('Unexpected VTL source anchor: '+old[:70])
    return text.replace(old,new)


def apply(directory):
    directory=Path(directory)
    header=directory/'GeometricGlottis2025.h'
    h=header.read_text('utf-8')
    h=replace_once(h,'#include "Glottis.h"','#include "Glottis.h"\n#include <cstdint>')
    h=replace_once(h,'  GeometricGlottis2025();','''  GeometricGlottis2025();
  void ptbPerturbation(bool enabled, double jitter, double shimmer, uint32_t seed, bool restart);
  double ptbShimmerFactor() const;
  static uint32_t ptbNext(uint32_t &state);
  bool ptbCustom = false;
  double ptbOriginalJitter = -1.0;
  double ptbShimmer = 0.0, ptbPreviousAmplitude = 1.0, ptbNextAmplitude = 1.0;
  uint32_t ptbSeed = 1, ptbJitterState = 1, ptbShimmerState = 2;
  long long ptbCycle = 0;
  void ptbResetRandom();''')
    header.write_text(h,encoding='utf-8')
    source=directory/'GeometricGlottis2025.cpp'
    s=source.read_text('utf-8')
    s=replace_once(s,'  phase = 0.0;','  phase = 0.0;\n  if (ptbCustom) ptbResetRandom();')
    if s.count('  amplitude *= relAmp;')!=2:raise ValueError('Unexpected amplitude sites')
    s=s.replace('  amplitude *= relAmp;','  amplitude *= relAmp;\n  if (ptbCustom && ptbShimmer > 0.0) amplitude *= ptbShimmerFactor();')
    s=replace_once(s,'if (rand() >= RAND_MAX / 2)',
        'if (ptbCustom ? (ptbNext(ptbJitterState) >= 0x80000000u) : (rand() >= RAND_MAX / 2))')
    s=replace_once(s,'  phase += 2 * M_PI * b * xi;','''  phase += 2 * M_PI * b * xi;
  if (ptbCustom && ptbShimmer > 0.0)
  {
    const long long cycle = (long long)floor(phase / (2.0 * M_PI));
    while (ptbCycle < cycle)
    {
      ++ptbCycle;
      ptbPreviousAmplitude = ptbNextAmplitude;
      ptbNextAmplitude = 1.0 + 0.2 * (ptbShimmer / 100.0) *
        (2.0 * (double)ptbNext(ptbShimmerState) / 4294967295.0 - 1.0);
    }
  }''')
    s+='''
// PhoneticToolbox opt-in source extension; GPL-3.0-or-later.
uint32_t GeometricGlottis2025::ptbNext(uint32_t &state)
{
  state ^= state << 13; state ^= state >> 17; state ^= state << 5;
  return state;
}
void GeometricGlottis2025::ptbResetRandom()
{
  ptbJitterState = ptbSeed ^ 0x9e3779b9u;
  ptbShimmerState = ptbSeed ^ 0x85ebca6bu;
  if (!ptbJitterState) ptbJitterState = 1;
  if (!ptbShimmerState) ptbShimmerState = 2;
  ptbCycle = 0;
  ptbPreviousAmplitude = 1.0;
  ptbNextAmplitude = 1.0 + 0.2 * (ptbShimmer / 100.0) *
    (2.0 * (double)ptbNext(ptbShimmerState) / 4294967295.0 - 1.0);
}
void GeometricGlottis2025::ptbPerturbation(bool enabled, double jitter, double shimmer, uint32_t seed, bool restart)
{
  if (ptbOriginalJitter < 0.0) ptbOriginalJitter = staticParams[JITTER].x;
  const bool reset = restart || enabled != ptbCustom || seed != ptbSeed;
  ptbCustom = enabled; ptbSeed = seed; ptbShimmer = shimmer;
  staticParams[JITTER].x = enabled ? jitter : ptbOriginalJitter;
  if (reset) ptbResetRandom();
}
double GeometricGlottis2025::ptbShimmerFactor() const
{
  const double position = phase / (2.0 * M_PI);
  const double fraction = position - floor(position);
  const double weight = 0.5 - 0.5 * cos(2.0 * M_PI * fraction / 2.0);
  return ptbPreviousAmplitude + (ptbNextAmplitude - ptbPreviousAmplitude) * weight;
}
'''
    source.write_text(s,encoding='utf-8')
    header=directory/'VocalTractLabApi.h'
    h=header.read_text('utf-8')
    h=replace_once(h,'#ifdef __cplusplus\n} /* end extern "C" */',
        'C_EXPORT int vtlSetSourcePerturbation(int enabled, double jitter, double shimmer, unsigned int seed, int restart);\n\n#ifdef __cplusplus\n} /* end extern "C" */')
    header.write_text(h,encoding='utf-8')
    source=directory/'VocalTractLabApi.cpp'
    s=source.read_text('utf-8')
    s+='''
// PhoneticToolbox source control extension. Existing ABI remains available.
int vtlSetSourcePerturbation(int enabled, double jitter, double shimmer, unsigned int seed, int restart)
{
  if (!vtlApiInitialized) return 1;
  if ((enabled != 0 && enabled != 1) || !std::isfinite(jitter) || !std::isfinite(shimmer) ||
      jitter < 0.0 || jitter > 100.0 || shimmer < 0.0 || shimmer > 100.0) return 2;
  auto *g = dynamic_cast<GeometricGlottis2025*>(glottises[selectedGlottis]);
  if (!g) return 3;
  g->ptbPerturbation(enabled != 0, jitter, shimmer, seed, restart != 0);
  auto *t = dynamic_cast<GeometricGlottis2025*>(tubeGlottis);
  if (t) t->ptbPerturbation(enabled != 0, jitter, shimmer, seed, restart != 0);
  if (enabled && restart) srand(seed);
  return 0;
}
'''
    s=replace_once(s,'#include <iostream>','#include <iostream>\n#include <cmath>\n#include <cstdlib>')
    source.write_text(s,encoding='utf-8')
