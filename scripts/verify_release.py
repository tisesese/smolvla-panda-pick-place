"""Check historical evidence and repository assets without GPU dependencies."""
from pathlib import Path
import ast
import json
import re
import xml.etree.ElementTree as ET
from summarize_results import summarize

ROOT=Path(__file__).resolve().parents[1]

def main():
    summary=summarize(check=True)
    for p in ROOT.rglob('*.py'):
        if '.git' not in p.parts and 'runs' not in p.parts:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    evidence=ROOT/'data_metadata/panda_100_lowdisk'
    split=json.loads((evidence/'split.json').read_text(encoding='utf-8'))
    train=split['train_episodes'];val=split['validation_episodes']
    assert len(train)==80 and len(val)==20 and not set(train)&set(val)
    manifest=json.loads((evidence/'manifest.json').read_text(encoding='utf-8'))
    lengths={x['episode']:x['frames'] for x in manifest['episodes']}
    assert sum(lengths.values())==136589
    plans=json.loads((evidence/'sampling-plans.json').read_text(encoding='utf-8'))
    for plan in plans.values():
        assert set(map(int,plan))==set(train)
        for ep,frames in plan.items():assert len(frames)==256 and all(0<=n<lengths[int(ep)] for n in frames)
    scene=ROOT/'simulation/experiments/panda_pick_place/scene/panda_pick_place.xml'
    root=ET.parse(scene).getroot();meshdir=scene.parent/root.find('compiler').get('meshdir')
    assert meshdir.is_dir()
    for mesh in root.findall('asset/mesh'):assert (meshdir/mesh.get('file')).is_file(),mesh.get('file')
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.git' in p.parts or 'runs' in p.parts:continue
        assert p.stat().st_size<90*1024**2,f'Unexpected large file: {p}'
        assert p.name!='model.safetensors' and p.suffix not in {'.parquet','.pt','.pth','.ckpt'},p
    print('PASS: 200 paired trials, fixed 80/20 split, three 20480-anchor plans, scene meshes, Python syntax and repository size boundaries')

if __name__=='__main__':main()
