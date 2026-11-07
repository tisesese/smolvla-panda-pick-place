"""Export step 6 evidence only after complete closed-loop admission."""
import csv
import json
import shutil
from pathlib import Path

from .config import EVIDENCE, OUTPUT
from .runtime import read,sha,write


def supporting_files(source, model, destination, selected, label):
    """Keep compact, portable evidence; leave the verified 1.1GB weights in WSL."""
    configuration=destination/"model_configuration"
    shutil.copytree(model,configuration,ignore=shutil.ignore_patterns("model.safetensors"))
    shutil.copy2(EVIDENCE/"train-only-stats.json",destination/"train-only-stats.json")
    protocols=[]
    current=model
    while True:
        lineage_path=current.parent/"lineage.json"
        if lineage_path.exists():
            lineage=read(lineage_path)
            training=OUTPUT/"training"/current.parents[2].name
            count=lineage["additional_steps"]
            parent=lineage["parent_steps"]
            previous=Path(lineage["source"])
        else:
            training=current.parents[2]
            count=read(current.parent/"manifest.json")["step"]
            parent=0
            previous=None
        protocol=read(training/"protocol.json")
        assert sha(EVIDENCE/"train-only-stats.json")==protocol.get("stats_sha256",protocol.get("train_stats_sha256"))
        updates=[json.loads(line) for line in (training/"updates.jsonl").read_text().splitlines()]
        updates=[row for row in updates if row["step"]<=count]
        assert len(updates)==count and [r["step"] for r in updates]==list(range(1,count+1))
        protocols.append((training,protocol,updates,parent,count))
        if previous is None:break
        current=previous
    curves=destination/"learning_curves"
    curves.mkdir()
    stages=[]
    for index,(training,protocol,updates,parent,count) in enumerate(reversed(protocols),1):
        stem=f"{index:02d}-{training.name}"
        write(curves/f"{stem}-protocol.json",protocol)
        for name in ["anchors.json","sampler.npy"]:
            shutil.copy2(training/name,curves/f"{stem}-{name}")
        with (curves/f"{stem}-updates.jsonl").open("w") as stream:
            stream.writelines(json.dumps(row)+"\n" for row in updates)
        csv_path=curves/f"{stem}-train.csv"
        with csv_path.open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(["local_step","lineage_step","train_flow_loss","trailing_100_mean"])
            losses=[]
            for row in updates:
                losses.append(row["loss"])
                writer.writerow([row["step"],parent+row["step"],row["loss"],sum(losses[-100:])/len(losses[-100:])])
        stages.append(dict(name=training.name,parent_steps=parent,actual_steps=count,
                           action_stride=protocol.get("action_stride",1),csv=csv_path.name))
    comparisons=read(source/"checkpoint-comparison.json")
    validation=[]
    for candidate in comparisons:
        steps=candidate["total_lineage_steps"]
        metrics=read(OUTPUT/"evaluations"/f"{source.name}_step{steps}_validation20_b4"/"metrics.json")
        assert metrics["samples"]==160 and metrics["action_stride"]==5 and metrics["batch_size"]==4
        assert metrics["model_sha256"]==read(Path(candidate["model"]).parent/"lineage.json")["weights_sha256"]
        write(curves/f"validation-{steps}.json",metrics)
        validation.append(dict(step=steps,flow_loss=metrics["mean_flow_loss"],
             joint_MAE_rad=metrics["joint_MAE_rad"],gripper_MAE=metrics["gripper_MAE"],
             closed_loop_successes=candidate["validation"]["successes"],
             closed_loop_attempts=candidate["validation"]["attempts"]))
    write(curves/"index.json",dict(stages=stages,validation=validation,
           note="Different temporal objectives are shown separately; teacher-forced loss is not closed-loop success."))
    with (curves/"validation.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(validation[0]));writer.writeheader();writer.writerows(validation)
    for row in selected["results"]:
        episode=row["episode"]
        evidence=OUTPUT/"rollouts"/label/f"episode{episode:03d}-K{row['execution_window']}"
        shutil.copytree(evidence,destination/"validation"/f"episode{episode:03d}")
    shutil.copy2(source/"checkpoint-comparison.json",destination/"checkpoint-comparison.json")
    audit=read(source/"independent-command-audit.json")
    assert audit["passed"]
    write(destination/"independent-command-audit.json",audit)
    checkpoint=OUTPUT/"training"/source.name/"checkpoints"/f"{read(model.parent/'lineage.json')['additional_steps']:06d}"
    for name in ["manifest.json","reload-verification.json"]:
        shutil.copy2(checkpoint/name,destination/name)
    shutil.copytree(source/"code",destination/"training_code_snapshot",ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(source/"code-sha256.json",destination/"training-code-sha256.json")
    weight= model/"model.safetensors"
    unc="\\\\wsl.localhost\\Ubuntu-24.04"+str(model).replace("/","\\")
    write(destination/"weights-location.json",dict(model_directory=model,windows_directory=unc,
          filename=weight.name,bytes=weight.stat().st_size,sha256=sha(weight),
          retained_in_existing_experiment=True,configuration_copy="model_configuration"))
    return validation[-1]


def export(name,destination):
    source=OUTPUT/"step6"/name
    selected=read(source/"selected.json")
    rows=selected["results"]
    assert selected["stage6_complete"] and len(rows)==5
    assert {r["episode"] for r in rows}=={32,33,34,35,36}
    assert all(r["pure_vla"] and r["selection_scope"]=="validation position" for r in rows)
    successful=[r for r in rows if r["success"]]
    assert len(successful)>=4
    for row in successful:
        assert row["ever_bilateral"] and row["ever_lifted"]
        assert row["reason"]=="independent_tray_stability_2s" and row["stable_seconds"]>=2.
        assert row["sim_seconds"]<=30.00001
    model=Path(selected["model"])
    lineage=read(model.parent/"lineage.json")
    assert sha(model/"model.safetensors")==lineage["weights_sha256"]
    destination=Path(destination)
    assert destination.is_absolute() and not destination.exists(),"Preserve prior exports"
    destination.mkdir(parents=True)
    write(destination/"selection.json",selected)
    write(destination/"lineage.json",lineage)
    write(destination/"training-variant.json",read(model/"training-variant.json"))
    label=selected.get("rollout_label",f"{name}_step{selected['total_lineage_steps']}")
    first=successful[0]
    horizon=first.get("execution_window",selected.get("execution_protocol",{}).get("execution_window",25))
    evidence=OUTPUT/"rollouts"/label/f"episode{first['episode']:03d}-K{horizon}"
    execution=read(evidence/"protocol.json")
    final_metrics=supporting_files(source,model,destination,selected,label)
    for original,copy in [("pure-vla.mp4","successful-validation.mp4"),("terminal.png","successful-terminal.png"),
                          ("protocol.json","validation-protocol.json")]:
        shutil.copy2(evidence/original,destination/copy)
    table="\n".join(f"| {r['episode']} | {'成功' if r['success'] else '失败'} | {r['sim_seconds']:.3f} | {r['stable_seconds']:.3f} | {r['reason']} |" for r in rows)
    precision="float32" if "float32"==execution.get("compute_precision") else "bfloat16"
    command=f".\\run.ps1 rollout --model {model} --episode {first['episode']} --horizon {horizon} --precision {precision} --label selected_validation_repeat_v1"
    if execution.get("flow_integration_steps") is not None:command+=f" --num-steps {execution['flow_integration_steps']}"
    if execution.get("gripper_command_gain",1.)!=1.:command+=f" --gripper-gain {execution['gripper_command_gain']}"
    if execution.get("gripper_calibration_knee",1.)!=1.:command+=f" --gripper-knee {execution['gripper_calibration_knee']}"
    if execution.get("max_gripper_increment",.06)!=.06:command+=f" --gripper-rate {execution['max_gripper_increment']}"
    if execution.get("independent_action_samples",1)!=1:command+=f" --action-samples {execution['independent_action_samples']}"
    text=f"""# 第六步：SmolVLA选模结果

已选定总计 **{selected['total_lineage_steps']:,} 步**的候选。固定5个验证位置完整成功 **{len(successful)}/5**，满足本工程4/5的开发门槛。该门槛不是PPT强制要求，也不代表总体成功率已达到80%。

模型目录：`{model}`

权重SHA256：`{lineage['weights_sha256']}`

本阶段从{lineage['parent_steps']:,}步权重续训{lineage['additional_steps']:,}步；原始正式训练为20,000步。训练只用原Train80；上述5个位置来自Validation20。模型输入为固定相机RGB、8维实测机器人状态和任务指令，抓取、搬运、下降、打开夹爪及撤退均由策略输出。固定图像裁剪和可训练模块见配套JSON。

成功判定沿用原仿真：方块抬升后搬入托盘，释放后独立稳定2秒，并在含初始静置的30秒内完成。原物理、接触力、速度和碰撞保护保留。

策略目标频率为{execution.get('predicted_target_hz',50)}Hz，原生控制频率仍为50Hz。相邻策略目标间的线性插值只使用已预测的动作，未使用额外物体信息。推理耗时另记，仿真在推理时暂停。

| 验证Episode | 完整结果 | 仿真秒 | 稳定秒 | 结束原因 |
|---|---|---:|---:|---|
{table}

成功视频：`successful-validation.mp4`，末帧：`successful-terminal.png`。全部逐步命令、图像和视频仍保存在WSL实验输出根目录。第八步50位置对照尚未启动。

第六步交付文件：`model_configuration/`含模型配置、预处理/反归一化参数及tokenizer；`train-only-stats.json`为原Train80统计量；`learning_curves/`含三个实际训练阶段的原始逐步损失、采样顺序、训练协议和五个候选的固定验证损失。权重原件保留在上述模型目录，Windows访问路径及校验值见`weights-location.json`，未重复占用C盘。`training_code_snapshot/`为实际训练运行的代码快照。`validation/`包含全部5次验证的视频、命令和原始动作块。

最终模型在Validation20的160个固定示教锚点上，flow loss为{final_metrics['flow_loss']:.6f}，关节MAE为{final_metrics['joint_MAE_rad']:.6f}rad，夹爪MAE为{final_metrics['gripper_MAE']:.6f}。这些是给定专家观测的离线误差，选模通过依据仍为4/5次完整闭环成功。此前四个候选各0/2提前淘汰，不能写作0/5。

成功用时范围为29.06—29.76仿真秒，距30秒上限余量较小。Episode33虽把方块搬到托盘，但未完成原生成功条件要求的充分张开与撤离，按超时失败记录。每次推理平均约{selected['validation']['inference_mean_s']:.3f}秒，单次验证实际墙钟约55—58秒；当前结果证明仿真闭环控制，尚不证明真实时间部署性能或正式独立测试成功率。

复现一个成功位置（使用新的label以保留原记录）：

```powershell
{command}
```
"""
    (destination/"第六步选模结果.md").write_text(text,encoding="utf-8")
    print(dict(exported=destination,successes=len(successful),attempts=5,model=str(model)),flush=True)
