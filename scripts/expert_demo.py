"""Run the original scripted expert; never count this as a VLA result."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from smolvla_task.runtime import setup
from smolvla_task.config import OLD,OUTPUT
sys.path.insert(0,str(OLD))

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);args=p.parse_args();setup()
    if not args.label or Path(args.label).name!=args.label:raise ValueError('Use one directory name as label')
    from experiments.panda_pick_place.expert import run
    result=run(OUTPUT/'expert-demos'/args.label,{'seed':20261023,'xyz_m':[.13,-.19,.782]},video=True)
    print({'scripted_expert_only':True,'success':result['success'],'sim_seconds':result['sim_seconds_including_reset']})

if __name__=='__main__':main()
