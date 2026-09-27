"""Explicit native raw/high-frame-rate capture; no inference in the capture path."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop/src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list-microphones',action='store_true')
    p.add_argument('--start',action='store_true',help='Explicitly open selected physical devices')
    p.add_argument('--camera-index',type=int,default=0)
    p.add_argument('--microphone-index',type=int)
    p.add_argument('--fps',type=float,default=60)
    p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.list_microphones:
        import sounddevice as sd
        print(json.dumps([dict(index=i,name=d['name'],channels=d['max_input_channels']) for i,d in enumerate(sd.query_devices()) if d['max_input_channels']],ensure_ascii=False));return
    if not args.start or args.output is None:p.error('--start and a NEW --output directory are required; devices remain closed')
    from ptb_desktop.m05_capture import record
    result=record(args.output,camera_index=args.camera_index,microphone_index=args.microphone_index,requested_fps=args.fps,duration=args.seconds)
    print(json.dumps(result,ensure_ascii=False));return 0 if result['complete'] else 1

if __name__=='__main__':raise SystemExit(main())
