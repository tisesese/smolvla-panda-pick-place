"""Controlled physical diagnostics; counterfactual replays are not policy scores."""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .config import EVIDENCE, OUTPUT
from .data import Demonstrations
from .environment import PandaTask
from .runtime import guard, read, setup, sha, write

OUT = OUTPUT/"step6/grasp_sensitivity_v1"


def launch():
    setup()
    guard()
    assert not OUT.exists(),"Preserve existing diagnostic attempts"
    code=OUT/"code"
    code.mkdir(parents=True)
    project=Path(__file__).resolve().parent.parent
    shutil.copytree(project/"smolvla_task",code/"smolvla_task",ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(project/"run_task.py",code/"run_task.py")
    write(OUT/"code-sha256.json",{str(p.relative_to(code)):sha(p) for p in code.rglob("*.py")})
    with (OUT/"diagnosis.log").open("a") as log:
        p=subprocess.Popen([sys.executable,"-u",str(code/"run_task.py"),"grasp-sensitivity"],cwd=code,
              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/"launch.json",dict(pid=p.pid,started_unix=time.time(),stage=6,log=OUT/"diagnosis.log"))
    print(read(OUT/"launch.json"),flush=True)


def replay(name,commands,position,gripper_gain=1.):
    begin=time.perf_counter()
    peak_clearance=-1.
    bilateral_samples=0
    with PandaTask() as env:
        env.reset(seed=position["seed"],options={"cube_xyz_m":position["xyz_m"]})
        for index,action in enumerate(commands):
            target=np.asarray(action,np.float32).astype(float).copy()
            desired=np.clip(1.-gripper_gain*(1.-target[7]),0.,1.)
            target[7]=np.clip(desired,env.native.previous[7]-.06,env.native.previous[7]+.06)
            # Exact recorded arm targets, with the original native limits,
            # tracking, force and collision checks retained on every physics step.
            _,_,done,truncated,info=env.native.step(target,include_rgb=False)
            m=env.native.last_info
            peak_clearance=max(peak_clearance,m["bottom_clearance_above_table_m"])
            bilateral_samples+=int(m["bilateral"])
            if done or truncated:break
        result=dict(name=name,diagnostic_counterfactual_replay=True,not_policy_success_evidence=True,
                    full_task_success=info["success"],reason=info["reason"],commands=index+1,
                    peak_clearance_m=peak_clearance,bilateral_control_samples=bilateral_samples,
                    gripper_gain=gripper_gain,reset_position=position,physical_peaks=env.native.peak,
                    wall_seconds=time.perf_counter()-begin)
    write(OUT/(name+".json"),result)
    print(name,dict(success=result["full_task_success"],peak_clearance_m=peak_clearance,reason=result["reason"]),flush=True)
    return result


def run():
    setup()
    try:
        ds=Demonstrations([0])
        original=read(EVIDENCE/"plan.json")["positions"][0]
        offsets=[(0,0),(.005,0),(-.005,0),(0,.005),(0,-.005)]
        write(OUT/"protocol.json",dict(stage=6,
              purpose="Distinguish grasp-position sensitivity from insufficient holding force",
              expert_xy_reset_offsets_m=offsets,model_recorded_command_gripper_gains=[1.1,1.2],
              no_policy_inference=True,not_model_selection_acceptance=True,not_final_test=True,
              safety_and_success_rules_unchanged=True))
        results={}
        for i,(dx,dy) in enumerate(offsets):
            position={**original,"xyz_m":[original["xyz_m"][0]+dx,original["xyz_m"][1]+dy,original["xyz_m"][2]]}
            key=f"expert-offset-{i}"
            results[key]=replay(key,ds.numeric[0]["action"],position)
            if i==0:assert results[key]["full_task_success"],"The original expert replay must still pass"
            write(OUT/"progress.json",dict(completed_cases=len(results),expected_cases=9,latest=key))
        for k in [25,50]:
            path=OUTPUT/f"rollouts/step6_20000_window{k}_pilot_v1/episode000-K{k}/commands.jsonl"
            rows=[json.loads(s) for s in path.read_text().splitlines()]
            commands=np.asarray([r["issued"] for r in rows],np.float32)
            for gain in [1.1,1.2]:
                key=f"recorded-window{k}-gain{gain}"
                results[key]=replay(key,commands,original,gripper_gain=gain)
                write(OUT/"progress.json",dict(completed_cases=len(results),expected_cases=9,latest=key))
        write(OUT/"complete.json",dict(completed=True,stage=6,cases=results,
              counterfactual_replay_only_not_new_model_success=True))
    except BaseException:
        write(OUT/"error.json",dict(traceback=traceback.format_exc(),unix_s=time.time()))
        raise

