"""Continuation after the 27k policy scored 32/50 complete tasks.

Development uses all 20 validation episodes. A fresh, immutable test is drawn
only after selecting weights and an executor. Prior test results are historical
development evidence; none are presented as fresh test results for this run.
"""
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .config import BASE, EVIDENCE, OUTPUT
from .data import Demonstrations, split
from .runtime import append, guard, read, setup, sha, write

SOURCE = OUTPUT / "step6/step6_stride5_vision_v1/candidates/027000/pretrained_model"
SOURCE_SHA = "c8bd09cc67a248c0c6c560ec6d5f3a853bef9cbe1935d5df8513764f0b75004a"
ADDITIONAL = 8000
PARENT = 27000
MILESTONES = [1000, 2000, 4000, 6000, 8000]
VALIDATION = [32, 37, 88, 33, 89, 38, 90, 34, 91, 39, 92, 35, 93, 94, 36, 95, 96, 97, 98, 99]
DEV_REQUIRED = 17


def directory(name):
    assert name and all(c.isalnum() or c in "_-" for c in name)
    return OUTPUT / "target80" / name


def training_name(name):
    return name + "_training"


def launch(name):
    setup(); guard()
    out = directory(name)
    assert not out.exists(), "Use a new run name; preserve all earlier results"
    assert sha(SOURCE / "model.safetensors") == SOURCE_SHA
    assert set(VALIDATION) == set(split()["validation_episodes"])
    code = out / "code"
    code.mkdir(parents=True)
    root = Path(__file__).resolve().parent.parent
    shutil.copytree(root / "smolvla_task", code / "smolvla_task", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(root / "run_task.py", code / "run_task.py")
    write(out / "code-sha256.json", {str(p.relative_to(code)): sha(p) for p in code.rglob("*.py")})
    protocol = dict(name=name, user_target="At least 80% COMPLETE tasks in a new independent 50-position evaluation",
        previous_primary_result=dict(steps=27000, successes=32, attempts=50, success_rate=.64),
        source=SOURCE, source_sha256=SOURCE_SHA, parent_steps=PARENT, additional_steps=ADDITIONAL,
        milestones=MILESTONES, variant="stride5_finish", peak_head_lr=3e-5, peak_tail_lr=3e-6,
        train_episodes=split()["train_episodes"], validation_episodes=VALIDATION,
        sampling="64 uniform +32 each approach,descend/grasp,lift,support,release,retract; Train80 only",
        development_gate=dict(required=DEV_REQUIRED, total=20, early_reject_after_failures=4),
        development_execution_windows=[10,5], action_stride=5, native_control_hz=50,
        input_whitelist=["RGB", "measured 8D robot state", "task instruction"],
        unchanged_timeout_sim_seconds=30, unchanged_stable_sim_seconds=2,
        unchanged_physics_and_safety=True, no_oracle_or_expert_action_control=True,
        final_test=dict(draw_only_after_selection=True, fresh_positions=50, required_successes=40,
                        execute_all_50=True, no_test_reselection=True,
                        interpretation="Observed sample success rate, not proof that population rate is >=80%"),
        reserve_prior_results=True)
    write(out / "protocol.json", protocol)
    from .selection_gate import register
    register(out,name)
    env = dict(os.environ, TASK4_REFINE_SOURCE=str(SOURCE), TASK4_REFINE_PARENT_STEPS=str(PARENT),
               TASK4_REFINE_ADDITIONAL_STEPS=str(ADDITIONAL))
    with (out / "pipeline.log").open("a") as log:
        process = subprocess.Popen([sys.executable,"-u",str(code/"run_task.py"),"target80-pipeline","--name",name],
            cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    write(out / "launch.json", dict(pid=process.pid, started_unix=time.time(), log=out/"pipeline.log", code=code,
                                    protocol_sha256=sha(out/"protocol.json")))
    print(read(out / "launch.json"), flush=True)


def validate(name, model, label, horizon):
    from .model import load, processors
    from .pipeline import aggregate
    from .rollout import run
    setup()
    assert read(directory(name)/"protocol.json")["validation_episodes"] == VALIDATION
    model = Path(model)
    policy = load(model)
    pre, post, _ = processors(policy, model)
    rows = []
    for episode in VALIDATION:
        completed=OUTPUT/"rollouts"/label/f"episode{episode:03d}-K{horizon}"/"result.json"
        if completed.exists():
            row=read(completed)
            assert row["model"]==str(model) and row["execution_window"]==horizon
        else:
            row=run(model, episode=episode, horizon=horizon, label=label, bundle=(policy,pre,post),
                noise_seed=20261023, scope="20-position development screening; never final test", num_steps=10)
        rows.append(row)
        failed = sum(not row["success"] for row in rows)
        result = dict(completed=len(rows), required=DEV_REQUIRED, expected=20, execution_window=horizon,
                      metrics=aggregate(rows), results=rows, early_rejected=failed>20-DEV_REQUIRED,
                      passed=len(rows)==20 and sum(r["success"] for r in rows)>=DEV_REQUIRED)
        write(OUTPUT/"rollouts"/label/"validation-summary.json", result)
        write(directory(name)/"validation-progress.json", dict(label=label, **result))
        if result["early_rejected"]:
            break


def deployment_check(out, selected):
    from .config import IMAGE, TASK
    from .model import batch, load, processors
    from .rollout import policy_sample
    model = Path(selected["model"])
    policy = load(model)
    pre, post, _ = processors(policy, model)
    for queue in policy._queues.values():
        queue.append(torch.zeros(8))
    policy.reset()
    assert all(len(q)==0 for q in policy._queues.values())
    sample = Demonstrations([32], action_stride=5).sample(32,0)
    train = batch([sample],pre)
    obs = {IMAGE:(sample[IMAGE].permute(1,2,0).numpy()*255).round().astype(np.uint8),
           "observation.state":sample["observation.state"].numpy(), "task":TASK}
    live = batch([policy_sample(obs)],pre)
    keys = [key for key,value in live.items() if torch.is_tensor(value)]
    for key in keys:
        torch.testing.assert_close(train[key],live[key],rtol=0,atol=0)
    error = float((post(train["action"]).cpu()-sample["action"].unsqueeze(0)).abs().max())
    assert error<1e-6
    write(out/"deployment-verification.json",dict(passed=True, weights_sha256=sha(model/"model.safetensors"),
        reset_clears_policy_queues=True, identical_train_deploy_tensor_keys=keys, denormalization_max_error=error,
        execution_window=selected["execution_window"], strict_checkpoint_reload_verified=True))


def pipeline(name):
    from .pipeline import child
    out = directory(name)
    if (out/"validation-protocol-v2.json").exists():
        from .selection_gate import pipeline as strengthened_pipeline
        return strengthened_pipeline(name)
    try:
        assert os.environ["TASK4_REFINE_SOURCE"] == str(SOURCE)
        assert int(os.environ["TASK4_REFINE_PARENT_STEPS"]) == PARENT
        assert int(os.environ["TASK4_REFINE_ADDITIONAL_STEPS"]) == ADDITIONAL
        import fcntl
        lock = (out/"pipeline.lock").open("a")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(out/"status.json",dict(action="gradient and memory preflight",target80_achieved=False))
        child("refinement-benchmark","--variant","stride5_finish")
        comparisons = []
        for milestone in MILESTONES:
            write(out/"status.json",dict(action="continuation training",lineage_target=PARENT+milestone,
                additional_target=milestone,target80_achieved=False))
            child("refine","--name",training_name(name),"--until",milestone,"--variant","stride5_finish")
            checkpoint=OUTPUT/"training"/training_name(name)/"checkpoints"/f"{milestone:06d}"
            child("verify-checkpoint",checkpoint)
            candidate=out/"candidates"/f"{PARENT+milestone:06d}"
            shutil.copytree(checkpoint/"pretrained_model",candidate/"pretrained_model",copy_function=os.link)
            model=candidate/"pretrained_model"
            write(candidate/"lineage.json",dict(parent_steps=PARENT,additional_steps=milestone,source=SOURCE,
                weights_sha256=sha(model/"model.safetensors"),strict_reload=read(checkpoint/"reload-verification.json")))
            for horizon in [10,5]:
                label=f"{name}_step{PARENT+milestone}_K{horizon}_dev20"
                write(out/"status.json",dict(action="20-position development validation",steps=PARENT+milestone,
                    execution_window=horizon,label=label,target80_achieved=False))
                child("validate-target80","--name",name,"--model",model,"--label",label,"--horizon",horizon)
                comparison=read(OUTPUT/"rollouts"/label/"validation-summary.json")
                record=dict(steps=PARENT+milestone,model=model,label=label,execution_window=horizon,validation=comparison)
                comparisons.append(record)
                write(out/"checkpoint-comparison.json",comparisons)
                if comparison["passed"]:
                    write(out/"selected.json",dict(**record,weights_sha256=sha(model/"model.safetensors"),
                        selected_on_development_only=True,final_test_not_yet_drawn=True,target80_achieved=False))
                    write(out/"status.json",dict(action="deployment input and strict reload verification",target80_achieved=False))
                    deployment_check(out,read(out/"selected.json"))
                    gc.collect()
                    torch.cuda.empty_cache()
                    child("test-target80","--name",name)
                    result=read(out/"independent50/complete.json")
                    write(out/"status.json",dict(action="new independent test finished",**result))
                    return
        write(out/"status.json",dict(action="bounded continuation evaluated; further diagnosis required",
            target80_achieved=False, no_new_final_test_run=True, latest_lineage_steps=PARENT+ADDITIONAL))
    except Exception as exc:
        write(out/"error.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        write(out/"status.json",dict(action="stopped with error; retain all evidence",target80_achieved=False))
        raise


def test(name):
    from .heldout import fresh_positions, summarize
    from .model import load, processors
    from .rollout import run
    setup(); guard()
    parent=directory(name)
    selected=read(parent/"selected.json")
    assert selected["validation"]["passed"] and selected["validation"]["completed"]==20
    from .selection_gate import passed
    expanded=selected.get("expanded_validation",{})
    assert expanded.get("passed") and expanded.get("completed")==100
    assert passed(expanded["results"]),"Selection must pass both complete 50-position rounds before steps 7/8"
    assert expanded["development_protocol_sha256"]==sha(parent/"development50/protocol.json")
    assert read(parent/"deployment-verification.json")["passed"]
    model=Path(selected["model"])
    assert sha(model/"model.safetensors")==selected["weights_sha256"]
    out=parent/"independent50"
    assert not out.exists(),"Never retest or overwrite the selected model on the final sample"
    out.mkdir()
    # The final sample is generated only after checkpoint selection has ended.
    positions=fresh_positions(out,sampling_seed=32610241,seed_base=742610240)
    protocol=dict(model=model,weights_sha256=selected["weights_sha256"],steps=selected["steps"],
        selection_sha256=sha(parent/"selected.json"),positions=positions,required_successes=40,attempts=50,
        execution_window=selected["execution_window"],action_stride=5,flow_steps=10,
        timeout_sim_seconds=30,stable_sim_seconds=2,policy_noise_seed_base=52610240,
        all_prior_test_positions_excluded=True,no_selection_on_this_test=True,scene_light_scale=1.)
    write(out/"protocol.json",protocol)
    policy=load(model);pre,post,_=processors(policy,model)
    rows=[]
    label=f"{name}_frozen_independent50"
    for index,position in enumerate(positions):
        row=run(model,horizon=selected["execution_window"],label=label,position=position,
            bundle=(policy,pre,post),noise_seed=52610240+index,num_steps=10,
            scope="New independent final test; never used to select this model")
        row.update(index=index,position=position,category=position["category"])
        rows.append(row);append(out/"results.jsonl",row)
        write(out/"progress.json",dict(completed=len(rows),expected=50,successes=sum(r["success"] for r in rows),
            current_sample_rate=sum(r["success"] for r in rows)/len(rows),target80_achieved=False))
    assert sha(model/"model.safetensors")==protocol["weights_sha256"]
    successes=sum(r["success"] for r in rows)
    result=dict(target80_achieved=successes>=40,successes=successes,attempts=50,success_rate=successes/50,
                selected_steps=selected["steps"],execution_window=selected["execution_window"],
                weights_sha256=selected["weights_sha256"],metrics=summarize(rows),
                full_sample_completed=True,unchanged_weights=True,
                statistical_scope="Observed 50-position sample; no claim of a population lower bound")
    write(out/"complete.json",result)
    print(result,flush=True)


def status(name):
    out=directory(name)
    result={"name":name}
    for key,path in [("launch",out/"launch.json"),("status",out/"status.json"),
        ("training",OUTPUT/"training"/training_name(name)/"progress.json"),
        ("validation",out/"validation-progress.json"),("final_test",out/"independent50/progress.json"),
        ("expanded_validation",out/"expanded-validation-progress.json"),
        ("complete",out/"independent50/complete.json"),("error",out/"error.json")]:
        if path.exists():
            value=read(path)
            if key in ["validation","expanded_validation"]:
                value={k:v for k,v in value.items() if k!="results"}
            result[key]=value
    if "launch" in result:
        if (out/"active-launch.json").exists():result["launch"]=read(out/"active-launch.json")
        pid=result["launch"]["pid"]
        state=Path(f"/proc/{pid}/status")
        cmd=Path(f"/proc/{pid}/cmdline")
        matched=state.exists() and cmd.exists() and name.encode() in cmd.read_bytes()
        state_value=next((x.split(":",1)[1].strip() for x in state.read_text().splitlines() if x.startswith("State:")),"") if matched else "exited"
        result["scheduler_process"]=dict(pid=pid,state=state_value,running=bool(matched and not state_value.startswith(("T","Z"))))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
