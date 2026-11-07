"""Sufficient development validation before deployment or the final test.

The existing training stream is preserved. Revisioned coordinator snapshots
make the stronger selection gate auditable without changing running training.
"""
import gc
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from .config import OUTPUT
from .runtime import append, guard, read, setup, sha, write


def register(out, name):
    from .heldout import fresh_positions
    path=out/"development50/protocol.json"
    if path.exists():
        return read(path)
    (out/"development50").mkdir()
    salt=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%100000
    positions=fresh_positions(out/"development50",sampling_seed=32610251+salt,seed_base=900000000+salt*100)
    protocol=dict(revision=2,reason="User requires sufficient selection validation >=80% before steps 7/8",
        positions=positions,unique_positions=50,noise_rounds=2,total_trials=100,
        required_successes_each_round=40,required_successes_total=80,
        selection_stage_only=True,not_the_final_test=True,
        round_noise_seed_bases=[62610250,62611250],
        early_reject="More than 10 failures in either round makes its 40/50 target impossible",
        stage1_screen=dict(required=17,total=20),execute_all_100_for_passing_candidates=True,
        downstream_gate="Both COMPLETE 50-trial rounds >=40 successes; then deployment and a separate fresh final test",
        unchanged_physics_safety_timeout_and_success=True,
        no_population_success_rate_guarantee=True)
    write(path,protocol)
    write(out/"validation-protocol-v2.json",dict(revision=2,development_protocol=path,
        development_protocol_sha256=sha(path),supersedes="20-position screen alone may not select a model",
        original_training_and_launch_protocol_sha256=sha(out/"protocol.json"),
        unchanged_training_protocol_sha256=sha(OUTPUT/"training"/(name+"_training")/"protocol.json")
            if (OUTPUT/"training"/(name+"_training")/"protocol.json").exists() else None,
        user_instruction="选模阶段验证充足并达到80%完整任务成功率，再进行后面的工作"))
    return protocol


def strengthen(name):
    from .target80 import directory, SOURCE, PARENT, ADDITIONAL
    setup();guard()
    out=directory(name)
    hold=read(out/"coordinator-gate-change.json")
    old_pid=hold["coordinator_pid"]
    command=Path(f"/proc/{old_pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
    assert f"target80-pipeline --name {name}" in command,command
    assert "T (stopped)" in Path(f"/proc/{old_pid}/status").read_text()
    assert not (out/"selected.json").exists() and not (out/"independent50").exists()
    revision=out/"revisions/002"
    code=revision/"code"
    assert not revision.exists(),"Preserve coordinator revisions"
    code.mkdir(parents=True)
    root=Path(__file__).resolve().parent.parent
    shutil.copytree(root/"smolvla_task",code/"smolvla_task",ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(root/"run_task.py",code/"run_task.py")
    write(revision/"code-sha256.json",{str(p.relative_to(code)):sha(p) for p in code.rglob("*.py")})
    protocol=register(out,name)
    # Terminate and resume only the stopped coordinator; the training child is
    # deliberately not signaled, so it can finish its current 1,000-step target.
    os.kill(old_pid,signal.SIGTERM)
    os.kill(old_pid,signal.SIGCONT)
    env=dict(os.environ,TASK4_REFINE_SOURCE=str(SOURCE),TASK4_REFINE_PARENT_STEPS=str(PARENT),
             TASK4_REFINE_ADDITIONAL_STEPS=str(ADDITIONAL))
    with (revision/"pipeline.log").open("a") as log:
        process=subprocess.Popen([sys.executable,"-u",str(code/"run_task.py"),"target80-pipeline","--name",name],
            cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    record=dict(pid=process.pid,started_unix=time.time(),revision=2,log=revision/"pipeline.log",code=code,
        superseded_coordinator_pid=old_pid,continuing_training_pid=hold["trainer_pid"],
        training_not_restarted=True,validation_protocol_sha256=sha(out/"validation-protocol-v2.json"))
    write(revision/"launch.json",record)
    write(out/"active-launch.json",record)
    print(record,flush=True)


def passed(rows):
    counts=[sum(r["success"] for r in rows if r["validation_round"]==round_id) for round_id in range(2)]
    attempts=[sum(r["validation_round"]==round_id for r in rows) for round_id in range(2)]
    return len(rows)==100 and attempts==[50,50] and all(n>=40 for n in counts)


def validate(name, model, label, horizon):
    from .model import load,processors
    from .pipeline import aggregate
    from .rollout import run
    from .target80 import directory
    setup()
    out=directory(name)
    protocol=read(out/"development50/protocol.json")
    assert len(protocol["positions"])==50 and protocol["noise_rounds"]==2
    model=Path(model)
    weights_sha=sha(model/"model.safetensors")
    policy=load(model);pre,post,_=processors(policy,model)
    rows=[]
    for round_id,noise_base in enumerate(protocol["round_noise_seed_bases"]):
        for index,position in enumerate(protocol["positions"]):
            round_label=f"{label}_round{round_id+1}"
            path=OUTPUT/"rollouts"/round_label/f"seed{position['seed']}"/"result.json"
            if path.exists():
                row=read(path)
                assert row["model"]==str(model) and row["execution_window"]==horizon
            else:
                row=run(model,horizon=horizon,label=round_label,position=position,bundle=(policy,pre,post),
                    noise_seed=noise_base+index,num_steps=10,
                    scope="Expanded selection-stage validation, not final test")
            row.update(validation_round=round_id,position_index=index,position=position,category=position["category"])
            rows.append(row)
            # A resumed validator rebuilds this summary from completed immutable
            # per-trial results rather than appending duplicate journal entries.
            counts=[sum(r["success"] for r in rows if r["validation_round"]==k) for k in range(2)]
            attempts=[sum(r["validation_round"]==k for r in rows) for k in range(2)]
            failed=attempts[round_id]-counts[round_id]
            result=dict(completed=len(rows),expected=100,unique_positions=50,required_each_round=40,
                round_attempts=attempts,round_successes=counts,metrics=aggregate(rows),results=rows,
                execution_window=horizon,weights_sha256=weights_sha,passed=passed(rows),early_rejected=failed>10,
                selection_stage_only=True,development_protocol_sha256=sha(out/"development50/protocol.json"))
            write(out/"expanded-validation-progress.json",dict(label=label,**result))
            write(OUTPUT/"rollouts"/label/"expanded-summary.json",result)
            if failed>10:
                return
    assert sha(model/"model.safetensors")==weights_sha


def pipeline(name):
    from .pipeline import child
    from .target80 import directory,training_name,PARENT,MILESTONES,deployment_check
    out=directory(name)
    try:
        assert int(os.environ["TASK4_REFINE_PARENT_STEPS"])==PARENT
        assert int(os.environ["TASK4_REFINE_ADDITIONAL_STEPS"])==8000
        import fcntl
        lock=(out/"pipeline.lock").open("a")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        revision=read(out/"validation-protocol-v2.json")
        assert sha(out/"development50/protocol.json")==revision["development_protocol_sha256"]
        if (out/"coordinator-gate-change.json").exists():
            trainer=read(out/"coordinator-gate-change.json")["trainer_pid"]
            write(out/"status.json",dict(action="continuation training under strengthened selection gate",revision=2,
                training_pid=trainer,training_not_restarted=True,target80_achieved=False,steps7_8_enabled=False))
            while Path(f"/proc/{trainer}/cmdline").exists():
                command=Path(f"/proc/{trainer}/cmdline").read_bytes()
                if training_name(name).encode() not in command:
                    break
                time.sleep(5)
        comparisons=read(out/"checkpoint-comparison.json") if (out/"checkpoint-comparison.json").exists() else []
        training=OUTPUT/"training"/training_name(name)
        for milestone in MILESTONES:
            latest=read(training/"latest.json") if (training/"latest.json").exists() else {"step":0}
            if latest["step"]<milestone:
                write(out/"status.json",dict(action="continuation training",revision=2,lineage_target=PARENT+milestone,
                    additional_target=milestone,target80_achieved=False,steps7_8_enabled=False))
                child("refine","--name",training_name(name),"--until",milestone,"--variant","stride5_finish")
            checkpoint=training/"checkpoints"/f"{milestone:06d}"
            if not checkpoint.exists():
                continue
            if not (checkpoint/"reload-verification.json").exists():
                child("verify-checkpoint",checkpoint)
            candidate=out/"candidates"/f"{PARENT+milestone:06d}"
            model=candidate/"pretrained_model"
            if not candidate.exists():
                shutil.copytree(checkpoint/"pretrained_model",model,copy_function=os.link)
                write(candidate/"lineage.json",dict(parent_steps=PARENT,additional_steps=milestone,
                    weights_sha256=sha(model/"model.safetensors"),strict_reload=read(checkpoint/"reload-verification.json")))
            for horizon in [10,5]:
                label=f"{name}_step{PARENT+milestone}_K{horizon}_dev20"
                summary=OUTPUT/"rollouts"/label/"validation-summary.json"
                write(out/"status.json",dict(action="20-position development screening",revision=2,steps=PARENT+milestone,
                    execution_window=horizon,label=label,target80_achieved=False,steps7_8_enabled=False))
                if not summary.exists() or not (read(summary)["completed"]==20 or read(summary)["early_rejected"]):
                    child("validate-target80","--name",name,"--model",model,"--label",label,"--horizon",horizon)
                screen=read(summary)
                record=dict(steps=PARENT+milestone,model=model,label=label,execution_window=horizon,validation=screen)
                if screen["passed"]:
                    expanded_label=f"{name}_step{PARENT+milestone}_K{horizon}_dev50x2"
                    write(out/"status.json",dict(action="50-position two-round development validation",revision=2,
                        steps=PARENT+milestone,execution_window=horizon,label=expanded_label,
                        target80_achieved=False,steps7_8_enabled=False))
                    child("validate-target80-expanded","--name",name,"--model",model,"--label",expanded_label,"--horizon",horizon)
                    record["expanded_validation"]=read(OUTPUT/"rollouts"/expanded_label/"expanded-summary.json")
                comparisons.append(record)
                write(out/"checkpoint-comparison.json",comparisons)
                if record.get("expanded_validation",{}).get("passed",False):
                    selected=dict(**record,weights_sha256=sha(model/"model.safetensors"),
                        selected_on_development_only=True,stage6_complete=True,selection_target80_verified=True,
                        final_test_not_yet_drawn=True,target80_achieved=False,selection_gate_revision=2)
                    write(out/"selected.json",selected)
                    write(out/"status.json",dict(action="sufficient selection validation passed; deployment verification",revision=2,
                        selection_target80_verified=True,target80_achieved=False,steps7_8_enabled=True))
                    deployment_check(out,selected)
                    gc.collect();torch.cuda.empty_cache()
                    child("test-target80","--name",name)
                    write(out/"status.json",dict(action="new independent test finished",revision=2,
                        selection_target80_verified=True,**read(out/"independent50/complete.json")))
                    return
        write(out/"status.json",dict(action="bounded continuation evaluated; further diagnosis required",revision=2,
            target80_achieved=False,selection_target80_verified=False,steps7_8_enabled=False,
            no_new_final_test_run=True,latest_lineage_steps=PARENT+max(MILESTONES)))
    except Exception as exc:
        write(out/"error.json",dict(error=repr(exc),traceback=traceback.format_exc(),revision=2))
        write(out/"status.json",dict(action="stopped with error; retain all evidence",target80_achieved=False,steps7_8_enabled=False))
        raise
