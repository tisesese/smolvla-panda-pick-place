"""Step 7 checks and frozen Step 8 evaluation of the selected 27k model."""
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

from .config import BASE,EVIDENCE,IMAGE,OLD,OUTPUT,TASK
from .runtime import append,guard,read,setup,sha,write

SELECTION="step6_stride5_vision_v1"
DEPLOYMENT=OUTPUT/"deployment"/"selected27000_v1"


def selection():
    selected=read(OUTPUT/"step6"/SELECTION/"selected.json")
    assert selected["stage6_complete"] and len(selected["results"])==5
    assert sum(x["success"] for x in selected["results"])>=4
    model=Path(selected["model"])
    assert sha(model/"model.safetensors")==read(model.parent/"lineage.json")["weights_sha256"]
    return selected,model


def executor_hashes():
    root=Path(__file__).parent
    return {name:sha(root/name) for name in ["environment.py","rollout.py","model.py","heldout.py"]}


def verify_deployment():
    from .data import Demonstrations
    from .environment import PandaTask
    from .model import batch,load,processors
    from .rollout import policy_sample
    from PIL import Image
    setup()
    assert not DEPLOYMENT.exists(),"Preserve completed deployment checks"
    DEPLOYMENT.mkdir(parents=True)
    selected,model=selection()
    p=load(model);pre,post,_=processors(p,model)
    for queue in p._queues.values():queue.append(torch.zeros(8))
    p.reset();assert all(len(q)==0 for q in p._queues.values())
    sample=Demonstrations([32],action_stride=5).sample(32,0)
    training=batch([sample],pre)
    obs={IMAGE:(sample[IMAGE].permute(1,2,0).numpy()*255).round().astype(np.uint8),
         "observation.state":sample["observation.state"].numpy(),"task":TASK}
    deployment=batch([policy_sample(obs)],pre)
    keys=[k for k,v in deployment.items() if torch.is_tensor(v)]
    for key in keys:torch.testing.assert_close(training[key],deployment[key],rtol=0,atol=0)
    roundtrip=post(training["action"]).cpu()
    error=float((roundtrip-torch.stack([sample["action"]])).abs().max())
    assert error<1e-6
    position=read(EVIDENCE/"plan.json")["positions"][32]
    with PandaTask() as env:
        live,_=env.reset(seed=position["seed"],options={"cube_xyz_m":position["xyz_m"]})
        nominal=live[IMAGE].copy();state=live["observation.state"].copy()
        Image.fromarray(nominal).save(DEPLOYMENT/"nominal.png")
        command,_=env.limit_action(np.r_[state[:7]+.5,0.])
        assert np.max(np.abs(command[:7]-env.native.previous[:7]))<=.0300001
    with PandaTask(light_scale=.9) as env:
        live,_=env.reset(seed=position["seed"],options={"cube_xyz_m":position["xyz_m"]})
        np.testing.assert_array_equal(state,live["observation.state"])
        light_diff=float(np.abs(nominal.astype(float)-live[IMAGE].astype(float)).mean())
        assert light_diff>0
        Image.fromarray(live[IMAGE]).save(DEPLOYMENT/"dimmed.png")
    repeated=read(OUTPUT/"rollouts/selected27000_reload_validation_v1/episode032-K10/result.json")
    assert repeated["success"] and repeated["stable_seconds"]>=2
    proof=dict(passed=True,model=model,weights_sha256=sha(model/"model.safetensors"),
        policy_reset_clears_all_action_queues=True,identical_training_deployment_tensor_keys=keys,
        action_denormalization_roundtrip_max_error=error,rate_limit_check_passed=True,
        prediction_targets_hz=10,native_control_hz=50,replan_every_sim_seconds=1,
        actual_light_scale_test=.9,lighting_image_mean_absolute_difference=light_diff,
        lighting_does_not_change_reset_robot_state=True,standalone_reload_success=repeated,
        selection_successes=selected["validation"]["successes"],executor_sha256=executor_hashes())
    write(DEPLOYMENT/"verification.json",proof)
    print({k:v for k,v in proof.items() if k not in ["standalone_reload_success","executor_sha256"]},flush=True)


def fresh_positions(out,sampling_seed=32610230,seed_base=732610230):
    """Register old published positions before drawing one immutable new set."""
    paths={EVIDENCE/"plan.json"}
    for root in [OLD/"outputs",OUTPUT]:
        for pattern in ["*protocol*.json","plan.json","registry.json"]:
            paths.update(p for p in root.rglob(pattern) if "/code/" not in str(p) and out not in p.parents)
    excluded=[];used_seeds=set();sources=[]
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ["xyz_m","cube_xyz_m","position_m"] and isinstance(item,list) and len(item)==3 and all(isinstance(x,(int,float)) for x in item):
                    excluded.append(np.asarray(item[:2],float))
                elif key=="seed" and isinstance(item,int):used_seeds.add(item)
                else:visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    for path in sorted(paths):
        visit(read(path));sources.append(dict(path=path,sha256=sha(path)))
    excluded=np.unique(np.asarray(excluded),axis=0)
    assert len(excluded)>=100
    rng=np.random.default_rng(sampling_seed);positions=[];minimum=[]
    for index in range(50):
        while True:
            x,y=rng.uniform(.105,.155),rng.uniform(-.215,-.165)
            if index>=40:
                edge=(index-40)%4
                if edge==0:x=.105
                elif edge==1:x=.155
                elif edge==2:y=-.215
                else:y=-.165
            xy=np.asarray([x,y]);distance=float(np.linalg.norm(excluded-xy,axis=1).min())
            if distance>1e-6 and all(np.linalg.norm(xy-np.asarray(p["xyz_m"][:2]))>1e-6 for p in positions):break
        seed=seed_base+index;assert seed not in used_seeds
        positions.append(dict(seed=seed,xyz_m=[float(x),float(y),.782],category="boundary" if index>=40 else "interior"))
        minimum.append(distance)
    write(out/"position-exclusion-audit.json",dict(excluded_unique_xy=len(excluded),sources=sources,
          sampling_seed=sampling_seed,minimum_distance_to_registered_old_positions_m=min(minimum),
          all_new_xy_and_seeds=True,exact_duplicate_tolerance_m=1e-6))
    return positions


def launch(name):
    setup();guard();selected,model=selection()
    proof=read(DEPLOYMENT/"verification.json")
    assert proof["passed"] and proof["executor_sha256"]==executor_hashes()
    out=OUTPUT/"heldout"/name
    assert not out.exists(),"Preserve all prior tests and their protocols"
    out.mkdir(parents=True)
    # Zero optimizer updates. Match every inference interface to isolate weights.
    baseline=out/"base_matched_model"
    shutil.copytree(model,baseline,copy_function=os.link,ignore=shutil.ignore_patterns("model.safetensors"))
    os.link(BASE/"model.safetensors",baseline/"model.safetensors")
    models=dict(finetuned27000=model,base=baseline)
    code=out/"code";code.mkdir()
    project=Path(__file__).resolve().parent.parent
    shutil.copytree(project/"smolvla_task",code/"smolvla_task",ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(project/"run_task.py",code/"run_task.py")
    write(out/"code-sha256.json",{str(p.relative_to(code)):sha(p) for p in code.rglob("*.py")})
    protocol=dict(stage=8,selection_record=OUTPUT/"step6"/SELECTION/"selected.json",training_steps=27000,
       model_directories=models,weights_sha256={k:sha(v/"model.safetensors") for k,v in models.items()},
       baseline_source=BASE,baseline_optimizer_updates=0,baseline_adaptation="Existing zero-update 8D task adapter; matched final crop, normalizers, dtype and temporal executor",
       positions=fresh_positions(out),light_stress_indices=[0,8,16,24,32,40,42,44,46,48],
       normal_light_scale=1.,stress_light_scale=.9,light_change="Scale actual MuJoCo light ambient/diffuse/specular; no pixel multiplier",
       initial_neutral_sim_seconds=1,timeout_sim_seconds=30,stable_sim_seconds=2,
       execution_window=10,action_stride=5,native_control_hz=50,replan_sim_seconds=1,
       compute_precision="bfloat16",flow_integration_steps=10,independent_action_samples=1,
       gripper_gain=1.,gripper_knee=1.,gripper_rate=.06,policy_noise_seed_base=42610230,
       primary_test_unique_seeds=50,total_trials=120,block_size=10,
       model_order="10-position blocks; alternate which model runs first; one GPU model resident",
       no_training_or_hyperparameter_selection_on_test=True,sim_paused_during_inference=True,
       inference_warmup_predictions_per_load=2,executor_sha256=executor_hashes(),
       metrics_definitions=dict(drop="Final cube below0.70m or native involuntary payload-loss termination; intended release excluded",
          collision="Any native robot-obstacle contact or collision termination; force-limit failures also tabulated separately",
          inference_latency="Preprocess + model + denormalization + CUDA synchronization; warmups and model loading excluded"))
    write(out/"protocol.json",protocol)
    with (out/"evaluation.log").open("a") as log:
        proc=subprocess.Popen([sys.executable,"-u",str(code/"run_task.py"),"heldout-pipeline","--name",name],
             cwd=code,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write(out/"launch.json",dict(pid=proc.pid,started_unix=time.time(),log=out/"evaluation.log",frozen_protocol_sha256=sha(out/"protocol.json")))
    print(read(out/"launch.json"),flush=True)


def summarize(rows):
    from .pipeline import aggregate
    result=aggregate(rows)
    latency=[x for row in rows for x in row["inference_latencies_seconds"]]
    result.update(drop_rate=result["dropped"]/len(rows),collision_rate=result["collisions"]/len(rows),
       mean_sim_time_all_trials_s=float(np.mean([r["sim_seconds"] for r in rows])),
       mean_wall_time_all_trials_s=float(np.mean([r["wall_seconds"] for r in rows])),
       inference_call_count=len(latency),inference_call_mean_s=float(np.mean(latency)),
       inference_call_median_s=float(np.median(latency)),inference_call_p95_s=float(np.percentile(latency,95)))
    return result


def metrics(rows):
    result={}
    for model in ["base","finetuned27000"]:
        for light in ["normal","dim"]:
            group=[r for r in rows if r["model_condition"]==model and r["lighting"]==light]
            if group:result[model+"-"+light]=summarize(group)
            for category in ["interior","boundary"]:
                subset=[r for r in group if r["category"]==category]
                if subset:result[model+"-"+light+"-"+category]=summarize(subset)
    return result


def pipeline(name):
    from .environment import PandaTask
    from .model import batch,load,processors
    from .rollout import policy_sample,run
    setup();out=OUTPUT/"heldout"/name
    try:
        protocol=read(out/"protocol.json")
        assert protocol["executor_sha256"]==executor_hashes()
        assert all(sha(Path(path)/"model.safetensors")==protocol["weights_sha256"][k] for k,path in protocol["model_directories"].items())
        rows=[];started=time.time()
        write(out/"status.json",dict(stage=8,action="running frozen paired test",completed_trials=0,expected_trials=120))
        for block_start in range(0,50,10):
            order=["finetuned27000","base"] if (block_start//10)%2==0 else ["base","finetuned27000"]
            for condition in order:
                guard();directory=Path(protocol["model_directories"][condition]);p=load(directory)
                pre,post,_=processors(p,directory)
                first=protocol["positions"][block_start]
                with PandaTask() as env:
                    obs,_=env.reset(seed=first["seed"],options={"cube_xyz_m":first["xyz_m"]})
                begin=time.perf_counter()
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    for _ in range(2):p.predict_action_chunk(batch([policy_sample(obs)],pre))
                torch.cuda.synchronize();p.reset()
                append(out/"warmups.jsonl",dict(block_start=block_start,model=condition,seconds=time.perf_counter()-begin))
                for index in range(block_start,block_start+10):
                    for light,scale in [("normal",1.)]+([("dim",.9)] if index in protocol["light_stress_indices"] else []):
                        position=protocol["positions"][index]
                        row=run(directory,horizon=10,label=f"{name}-{condition}-{light}",
                             noise_seed=protocol["policy_noise_seed_base"]+index,position=position,bundle=(p,pre,post),
                             scope="Frozen independent held-out paired test",light_scale=scale,num_steps=10)
                        row.update(model_condition=condition,lighting=light,index=index,category=position["category"])
                        rows.append(row);append(out/"results.jsonl",row)
                        write(out/"progress.json",dict(completed_trials=len(rows),expected_trials=120,
                              last_model=condition,last_index=index,last_lighting=light,metrics=metrics(rows),
                              elapsed_wall_seconds=time.time()-started))
                del p,pre,post;gc.collect();torch.cuda.empty_cache()
        assert len(rows)==120
        for condition in ["base","finetuned27000"]:
            assert {r["index"] for r in rows if r["model_condition"]==condition and r["lighting"]=="normal"}==set(range(50))
        write(out/"complete.json",dict(completed=True,training_steps=27000,unique_test_seeds=50,
              total_trials=120,metrics=metrics(rows),frozen_protocol_sha256=sha(out/"protocol.json"),
              elapsed_wall_seconds=time.time()-started,no_test_driven_model_changes=True))
        write(out/"status.json",dict(stage=8,action="completed",completed_trials=120,expected_trials=120))
        print("HELDOUT_COMPLETE",read(out/"complete.json"),flush=True)
    except BaseException:
        write(out/"error.json",dict(traceback=traceback.format_exc(),unix_s=time.time()))
        write(out/"status.json",dict(stage=8,action="stopped with error"))
        raise


def status(name):
    out=OUTPUT/"heldout"/name
    result={key:read(out/f"{key}.json") for key in ["launch","status","progress","complete","error"] if (out/f"{key}.json").exists()}
    if "launch" in result:
        pid=result["launch"]["pid"];state=Path(f"/proc/{pid}/status")
        result["process_exists"]=state.exists()
        if state.exists():result["process_state"]=next(x for x in state.read_text().splitlines() if x.startswith("State:"))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
