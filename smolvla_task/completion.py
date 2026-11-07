"""Complete the frozen Base/light comparison after the new fine-model test."""
import gc
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from .config import BASE,EVIDENCE,OUTPUT,SCENE,OLD
from .runtime import append,guard,read,setup,sha,write
from .target80 import directory


def metrics(rows,fine):
    from .heldout import summarize
    output={}
    for model in ['base',fine]:
        for light in ['normal','dim']:
            group=[r for r in rows if r['model_condition']==model and r['lighting']==light]
            if group:output[model+'-'+light]=summarize(group)
            for category in ['interior','boundary']:
                subset=[r for r in group if r['category']==category]
                if subset:output[model+'-'+light+'-'+category]=summarize(subset)
    return output


def launch(name):
    setup();guard()
    parent=directory(name);selected=read(parent/'selected.json')
    assert selected['expanded_validation']['passed'] and selected['expanded_validation']['completed']==100
    assert read(parent/'deployment-verification.json')['passed']
    formal=parent/'independent50';frozen=read(formal/'protocol.json')
    model=Path(selected['model']);steps=selected['steps'];fine=f'finetuned{steps}'
    assert sha(model/'model.safetensors')==frozen['weights_sha256']==selected['weights_sha256']
    out=parent/'comparison120';assert not out.exists(),'Retain all prior paired evaluations'
    code=out/'code';code.mkdir(parents=True)
    root=Path(__file__).resolve().parent.parent
    # Verify the controller matches the ongoing, already frozen fine test.
    source_code=parent/'revisions/002/code/smolvla_task'
    for filename in ['environment.py','rollout.py','model.py']:
        assert sha(root/'smolvla_task'/filename)==sha(source_code/filename)
    shutil.copytree(root/'smolvla_task',code/'smolvla_task',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(root/'run_task.py',code/'run_task.py')
    write(out/'code-sha256.json',{str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')})
    baseline=out/'base_matched_model'
    shutil.copytree(model,baseline,copy_function=os.link,ignore=shutil.ignore_patterns('model.safetensors'))
    os.link(BASE/'model.safetensors',baseline/'model.safetensors')
    proof=read(BASE.parent/'verified.json')
    assert proof['passed'] and proof['optimizer_updates']==0 and proof['all_official_Base_values_strict_equal']
    assert sha(baseline/'model.safetensors')==proof['model_sha256']
    protocol=dict(stage=8,training_steps=steps,fine_condition=fine,workspace_root=root,
        delivery_destination=root/'artifacts'/f'第八步完整测试结果_{steps}步_v2',
        model_directories={fine:model,'base':baseline},
        weights_sha256={fine:selected['weights_sha256'],'base':proof['model_sha256']},
        positions=frozen['positions'],light_stress_indices=[0,8,16,24,32,40,42,44,46,48],
        original_fine50_protocol=formal/'protocol.json',original_fine50_protocol_sha256=sha(formal/'protocol.json'),
        original_selection_sha256=sha(parent/'selected.json'),
        existing_fine50_is_not_rerun=True,primary_test_unique_seeds=50,total_trials=120,
        model_order='Frozen fine normal50, then Base normal50, Base dim10, fine dim10; one model resident',
        execution_window=selected['execution_window'],action_stride=5,native_control_hz=50,flow_steps=10,
        compute_precision='bfloat16',policy_noise_seed_base=frozen['policy_noise_seed_base'],
        normal_light_scale=1.,stress_light_scale=.9,brightness_multiplier=1.,gripper_gain=1.,gripper_rate=.06,
        timeout_sim_seconds=30,stable_sim_seconds=2,sim_paused_during_inference=True,
        no_test_driven_model_or_executor_changes=True,baseline_optimizer_updates=0,
        baseline_interface='Zero-update official Base with the selected task crop, normalizers and action-time interface',
        inference_latency_definition='All actual inference calls, including first calls; model loading excluded; no discarded warmups',
        executor_sha256={n:sha(root/'smolvla_task'/n) for n in ['environment.py','rollout.py','model.py']})
    write(out/'protocol.json',protocol)
    assets={str(path):sha(path) for path in [SCENE,EVIDENCE/'train-only-stats.json',EVIDENCE/'split.json',
        OLD/'experiments/panda_pick_place/environment.py']}
    write(out/'external-assets-and-runtime.json',dict(assets_sha256=assets,
        packages={n:metadata.version(n) for n in ['torch','lerobot','mujoco','gymnasium','transformers','safetensors']}))
    with (out/'pipeline.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(code/'run_task.py'),'completion-pipeline','--name',name],
            cwd=code,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write(out/'launch.json',dict(pid=process.pid,started_unix=time.time(),log=out/'pipeline.log',code=code,
        frozen_protocol_sha256=sha(out/'protocol.json')))
    print(read(out/'launch.json'),flush=True)


def pipeline(name):
    setup()
    parent=directory(name);out=parent/'comparison120'
    try:
        import fcntl
        lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=read(out/'protocol.json');fine=protocol['fine_condition']
        write(out/'status.json',dict(action='waiting for frozen fine50 to finish',completed_trials=0,expected_trials=120))
        formal=parent/'independent50'
        while not (formal/'complete.json').exists():
            if (parent/'error.json').exists():raise RuntimeError('Original fine-test pipeline reported an error')
            time.sleep(5)
        complete=read(formal/'complete.json')
        if not complete['target80_achieved']:
            write(out/'status.json',dict(action='formal fine50 below80; return to model development',
                target80_achieved=False,successes=complete['successes'],attempts=50))
            return
        # The fine test writes its result immediately before exiting; wait for
        # that GPU owner to exit before loading another policy.
        while True:
            active=[]
            for path in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    cmd=path.read_bytes()
                    if b'test-target80' in cmd and name.encode() in cmd:active.append(path)
                except OSError:pass
            if not active:break
            time.sleep(2)
        rows=[]
        for line in (formal/'results.jsonl').read_text().splitlines():
            row=json.loads(line)
            row.update(model_condition=fine,lighting='normal',category=protocol['positions'][row['index']]['category'],
                original_rollout=str(OUTPUT/'rollouts'/f'{name}_frozen_independent50'/f"seed{row['seed']}"))
            rows.append(row);append(out/'results.jsonl',row)
        assert len(rows)==50 and {r['index'] for r in rows}==set(range(50))
        write(out/'progress.json',dict(completed_trials=50,expected_trials=120,metrics=metrics(rows,fine)))
        started=time.time()
        from .model import load,processors
        from .rollout import run
        for condition,light,indices in [('base','normal',range(50)),('base','dim',protocol['light_stress_indices']),
                                        (fine,'dim',protocol['light_stress_indices'])]:
            guard();model=Path(protocol['model_directories'][condition]);policy=load(model)
            pre,post,_=processors(policy,model)
            for index in indices:
                write(out/'status.json',dict(action='running frozen Base/light comparison',model=condition,lighting=light,
                    index=index,completed_trials=len(rows),expected_trials=120,
                    label=f'{name}_comparison-{condition}-{light}',selection_target80_verified=True))
                position=protocol['positions'][index]
                row=run(model,horizon=protocol['execution_window'],label=f'{name}_comparison-{condition}-{light}',
                    position=position,bundle=(policy,pre,post),noise_seed=protocol['policy_noise_seed_base']+index,
                    scope='Frozen paired formal test, no model selection',num_steps=10,light_scale=1. if light=='normal' else .9)
                row.update(model_condition=condition,lighting=light,index=index,category=position['category'],
                    position=position,original_rollout=str(OUTPUT/'rollouts'/f'{name}_comparison-{condition}-{light}'/f"seed{position['seed']}"))
                rows.append(row);append(out/'results.jsonl',row)
                write(out/'progress.json',dict(completed_trials=len(rows),expected_trials=120,last_model=condition,
                    last_lighting=light,last_index=index,metrics=metrics(rows,fine),elapsed_wall_seconds=time.time()-started))
            del policy,pre,post;gc.collect();torch.cuda.empty_cache()
        assert len(rows)==120 and len({(r['model_condition'],r['lighting'],r['index']) for r in rows})==120
        for condition,path in protocol['model_directories'].items():
            assert sha(Path(path)/'model.safetensors')==protocol['weights_sha256'][condition]
        assert sha(out/'protocol.json')==read(out/'launch.json')['frozen_protocol_sha256']
        assets=read(out/'external-assets-and-runtime.json')
        assert all(sha(Path(p))==digest for p,digest in assets['assets_sha256'].items())
        write(out/'complete.json',dict(completed=True,training_steps=protocol['training_steps'],fine_condition=fine,
            total_trials=120,unique_test_seeds=50,metrics=metrics(rows,fine),target80_achieved=True,
            frozen_protocol_sha256=sha(out/'protocol.json'),no_test_driven_model_changes=True,
            original_fine50_results_sha256=sha(formal/'results.jsonl')))
        write(out/'status.json',dict(action='paired tests completed; exporting evidence',completed_trials=120,expected_trials=120))
        from .completion_export import export
        export(name)
        write(out/'status.json',dict(action='paired tests and evidence export completed',completed_trials=120,expected_trials=120))
    except Exception as exc:
        write(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix_s=time.time()))
        write(out/'status.json',dict(action='comparison stopped with error'))
        raise
