"""Recreate task-local tools from the verified backup, without global installs."""
import argparse,hashlib,json,os,platform,subprocess,sys
from pathlib import Path

def npm_command(npm,cache,*,allow_network=False):
    return [str(npm),'ci',*(['--prefer-offline'] if allow_network else ['--offline']),'--cache',str(cache),'--no-audit','--no-fund']

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--allow-npm-network',action='store_true',help='Allow npm ci to fill only this task npm cache from the current lockfile; Python remains offline');a=p.parse_args();root=a.root.resolve()
    if sys.platform!='darwin' or platform.machine()!='arm64':raise SystemExit('macOS arm64 required')
    uv=root/'tools/uv';source=root/'source'
    runtimes=[p for p in (root/'python').glob('*/bin/python3.11') if not p.parent.parent.is_symlink()];nodes=list((root/'tools').glob('node-*-darwin-arm64/bin'))
    if len(runtimes)!=1 or len(nodes)!=1:raise RuntimeError('Expected the backed-up Python and Node runtimes')
    env=os.environ.copy();env.update(UV_CACHE_DIR=str(root/'cache/uv'),UV_PYTHON_INSTALL_DIR=str(root/'python'),
        UV_DEFAULT_INDEX='https://pypi.tuna.tsinghua.edu.cn/simple',PYTHONDONTWRITEBYTECODE='1',PATH=str(nodes[0])+os.pathsep+env['PATH'])
    python=root/'envs/host/bin/python'
    if python.exists():raise RuntimeError('Choose a new restore folder; existing environment preserved')
    subprocess.run([str(uv),'venv','--python',str(runtimes[0]),str(python.parents[1])],env=env,check=True)
    base=[str(uv),'pip','install','--offline','--python',str(python)]
    subprocess.run(base+['-r',str(source/'requirements/requirements-macos-host.lock')],env=env,check=True)
    wheels=[root/'downloads'/name for name in ('numpy-2.2.6-cp311-cp311-macosx_11_0_arm64.whl','scipy-1.16.3-cp311-cp311-macosx_12_0_arm64.whl')]
    subprocess.run(base+['--reinstall','--no-deps',*map(str,wheels)],env=env,check=True)
    media=source/'resources/macos-media-provenance.json'
    if not media.is_file():raise RuntimeError('Namespaced PyAV provenance is required')
    if media.exists():
        receipt=json.loads(media.read_text());name=receipt['wheel']
        if Path(name).name!=name:raise RuntimeError('Invalid macOS media wheel name')
        wheel=root/'downloads'/name
        if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest()!=receipt['sha256']:
            raise RuntimeError('Required namespaced PyAV wheel missing or changed')
        subprocess.run(base+['--reinstall','--no-deps',str(wheel)],env=env,check=True)
    subprocess.run(base+['--no-deps','--no-build-isolation',*[str(source/n) for n in ('packages/phonetic_core','backend','desktop')]],env=env,check=True)
    subprocess.run(npm_command(nodes[0]/'npm',root/'cache/npm',allow_network=a.allow_npm_network),cwd=source/'frontend',env=env,check=True)
    print('Task environment restored; npm network allowed:',a.allow_npm_network)
if __name__=='__main__':main()
