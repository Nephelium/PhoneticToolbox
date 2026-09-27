"""Local M05 offline operation entry. Writes a NEW result directory only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'backend/src'), str(ROOT / 'packages/phonetic_core/src')]
from ptb_worker.m05_video import JsonLinesSink, analyze_video, sha256_file
from phonetic_core.lip.sequence import LipConfig
from ptb_worker.m05_results import export_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--no-filter', action='store_true')
    parser.add_argument('--cutoff', type=float, default=15.)
    parser.add_argument('--animation',choices=['none','mp4','gif'],default='none')
    parser.add_argument('--quality',choices=['high','standard','small'],default='standard')
    parser.add_argument('--offset',type=float,default=0.)
    args = parser.parse_args()
    if not -2<=args.offset<=2:parser.error('offset must be finite and within -2 to 2 seconds')
    if args.output.exists(): parser.error('Output must be a new directory; existing data will not be overwritten')
    args.output.mkdir(parents=True)
    result = args.output / 'frames.jsonl'
    try:
        with result.open('xb') as stream:
            sink = JsonLinesSink(stream, 512_000_000)
            manifest = analyze_video(args.video, sink, LipConfig(not args.no_filter, args.cutoff),
                                     progress=lambda p: print(json.dumps(p), flush=True))
        from ptb_worker.m05_audio import extract_audio
        manifest['audio'],audio_names=extract_audio(args.video,args.output)
        from ptb_worker.m05_alignment import suggest_offset
        manifest['offset_suggestion']=suggest_offset(args.output,manifest)
        names = [result.name,*audio_names, *export_tables(result, args.output, manifest)]
        if args.animation!='none':
            from ptb_worker.m05_animation import export_animation
            name='lip-animation.'+args.animation
            manifest['animation']=export_animation(result,args.output/name,manifest,quality=args.quality,format=args.animation,offset=args.offset,audio_path=args.output/'audio_recording.wav' if manifest['audio']['present'] else None)
            names.append(name)
        manifest['files'] = [dict(name=name, sha256=sha256_file(args.output/name), bytes=(args.output/name).stat().st_size) for name in names]
        with (args.output / 'manifest.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, allow_nan=False, indent=2)
        print(json.dumps(dict(success=True, output=str(args.output), frames=manifest['timing']['decoded_frames'])))
    except BaseException as error:
        (args.output / 'incomplete.json').write_text(json.dumps(dict(complete=False, error=type(error).__name__)), encoding='utf-8')
        raise


if __name__ == '__main__': main()
