
"""Portable paths and serialization for the original native task."""
import os
from pathlib import Path
import hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'experiments/panda_pick_place'
OUT=ROOT/'outputs/panda_pick_place'
SCENE=EXP/'scene/panda_pick_place.xml'
TASK='Pick up the red cube and place it in the tray.'
def convert(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,o):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(o,indent=2,default=convert,allow_nan=False)+'\n',encoding='utf-8')
def log(event,**kw):print(json.dumps({'event':event,'unix_s':time.time(),**kw},default=convert),flush=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
