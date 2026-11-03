"""Offline Ed25519 manifest signing with an existing key; no key generation/network."""
import argparse
import base64
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop/src'))
from ptb_desktop.update_signatures import manifest_payload,verify_manifest


def sign_manifest(manifest,key,key_id):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    if not isinstance(key,Ed25519PrivateKey):raise ValueError('An Ed25519 private key is required')
    if not isinstance(key_id,str) or not key_id.isascii() or not 1<=len(key_id)<=80 or not all(c.isalnum() or c in '-_.' for c in key_id):
        raise ValueError('Invalid key ID')
    if manifest.get('schemaVersion')!='ptb-release/1':raise ValueError('Sign each ptb-release/1 manifest, not an index')
    signed=dict(manifest)
    signed['signature']={'algorithm':'Ed25519','keyId':key_id,'value':base64.b64encode(key.sign(manifest_payload(manifest))).decode('ascii')}
    public=base64.b64encode(key.public_key().public_bytes_raw()).decode('ascii')
    verify_manifest(signed,{key_id:public})
    return signed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--key',type=Path,required=True,help='Existing PEM private key; never copied to output')
    parser.add_argument('--key-id',required=True)
    parser.add_argument('--password-stdin',action='store_true',help='Read encrypted-key password from one stdin line')
    args=parser.parse_args()
    if args.input.stat().st_size>1000000 or args.key.stat().st_size>65536:parser.error('Input is too large')
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    password=sys.stdin.buffer.readline(4097).rstrip(b'\r\n') if args.password_stdin else None
    key=load_pem_private_key(args.key.read_bytes(),password=password)
    signed=sign_manifest(json.loads(args.input.read_text('utf-8')),key,args.key_id)
    # Never overwrite a key, input, existing signed manifest or other file.
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(signed,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print('Signed manifest written and signature verified.')


if __name__=='__main__':main()
