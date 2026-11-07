"""Export a completed frozen paired evaluation, without editing its model."""
import csv
import json
import importlib.metadata as metadata
import shutil
from pathlib import Path

import numpy as np

from .config import OUTPUT
from .heldout import metrics
from .runtime import read,sha,write


def diagnose(row,commands):
    final=commands[-1]["evaluation_only"]
    if row["success"]:return "完整成功"
    if row["collision"]:return "碰撞或障碍接触"
    if row["dropped"]:return "搬运中掉落"
    if not row["ever_bilateral"]:return "未建立双指抓取"
    if not row["ever_lifted"]:return "未完成有效抬升"
    if not final["cube_within_interior"]:return "搬运或入盘未完成"
    if final["gripper_width_m"]<.075 or commands[-1]["issued"][7]<.98:return "入盘后张开夹爪不足"
    if row["stable_seconds"]<2:return "撤离或稳定时间不足"
    return "其他原生终止条件"


def export(name,destination):
    source=OUTPUT/"heldout"/name
    complete=read(source/"complete.json");protocol=read(source/"protocol.json")
    assert complete["completed"] and complete["total_trials"]==120 and complete["unique_test_seeds"]==50
    assert complete["frozen_protocol_sha256"]==sha(source/"protocol.json")==read(source/"launch.json")["frozen_protocol_sha256"]
    assert complete["no_test_driven_model_changes"]
    rows=[json.loads(line) for line in (source/"results.jsonl").read_text().splitlines()]
    assert len(rows)==120 and len({(r["model_condition"],r["lighting"],r["index"]) for r in rows})==120
    for model in ["base","finetuned27000"]:
        for light,indices in [("normal",set(range(50))),("dim",set(protocol["light_stress_indices"]))]:
            assert {r["index"] for r in rows if r["model_condition"]==model and r["lighting"]==light}==indices
        directory=Path(protocol["model_directories"][model])
        assert sha(directory/"model.safetensors")==protocol["weights_sha256"][model]
    external=read(source/"external-assets-and-runtime.json")
    assert all(sha(Path(path))==digest for path,digest in external["assets_sha256"].items())
    assert all(metadata.version(name)==version for name,version in external["packages"].items())
    assert metrics(rows)==complete["metrics"]
    destination=Path(destination)
    assert destination.is_absolute() and not destination.exists(),"Preserve previous exports"
    destination.mkdir(parents=True)
    installed=sorted({f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions() if dist.metadata.get("Name")})
    (destination/"runtime-installed-versions.txt").write_text("\n".join(installed)+"\n",encoding="utf-8")
    for name_ in ["protocol.json","complete.json","position-exclusion-audit.json","external-assets-and-runtime.json","results.jsonl","code-sha256.json"]:
        shutil.copy2(source/name_,destination/name_)
    shutil.copytree(source/"code",destination/"frozen_evaluation_code",ignore=shutil.ignore_patterns("__pycache__"))
    baseline_proof=Path(protocol["baseline_source"]).parent/"verified.json"
    proof=read(baseline_proof)
    assert proof["passed"] and proof["optimizer_updates"]==0 and proof["all_official_Base_values_strict_equal"]
    assert proof["model_sha256"]==protocol["weights_sha256"]["base"]
    shutil.copy2(baseline_proof,destination/"base-zero-update-verification.json")
    csv_rows=[];failure_counts={};latencies={};evidence=[]
    for row in rows:
        condition,light,index=row["model_condition"],row["lighting"],row["index"]
        pos=protocol["positions"][index];assert row["seed"]==pos["seed"]
        original=OUTPUT/"rollouts"/f"{name}-{condition}-{light}"/f"seed{pos['seed']}"
        execution=read(original/"protocol.json")
        assert execution["position"]==pos and execution["execution_window"]==10 and execution["action_stride"]==5
        assert execution["scene_light_scale"]==(1. if light=="normal" else .9)
        assert execution["image_brightness_multiplier"]==1 and execution["independent_action_samples"]==1
        commands=[json.loads(x) for x in (original/"commands.jsonl").read_text().splitlines()]
        assert len(commands)==row["commands"]
        if row["success"]:
            final=commands[-1]["evaluation_only"]
            assert row["ever_bilateral"] and row["ever_lifted"] and row["stable_seconds"]>=2 and row["sim_seconds"]<=30
            assert final["cube_within_interior"] and not final["cube_robot_contacts"] and final["tray_base_support_force_N"]>0 and final["gripper_width_m"]>=.075
        diagnosis=diagnose(row,commands)
        if not row["success"]:
            key=condition+"-"+light;failure_counts.setdefault(key,{})
            failure_counts[key][diagnosis]=failure_counts[key].get(diagnosis,0)+1
        relative=Path("videos")/condition/light/f"index{index:02d}-seed{pos['seed']}"
        target=destination/relative;target.mkdir(parents=True)
        for filename in ["pure-vla.mp4","terminal.png","result.json","protocol.json"]:
            shutil.copy2(original/filename,target/filename)
        evidence.append(dict(model=condition,lighting=light,index=index,success=row["success"],diagnosis=diagnosis,
             video=str(relative/"pure-vla.mp4"),commands_original=original/"commands.jsonl",raw_actions_original=original))
        latencies.setdefault(condition+"-"+light,[]).extend(row["inference_latencies_seconds"])
        csv_rows.append(dict(model=condition,lighting=light,index=index,seed=row["seed"],category=row["category"],
            cube_initial_x_m=pos["xyz_m"][0],cube_initial_y_m=pos["xyz_m"][1],success=row["success"],
            simulation_seconds=row["sim_seconds"],stable_seconds=row["stable_seconds"],wall_seconds=row["wall_seconds"],
            inference_mean_seconds=row["inference_mean_seconds"],inference_p95_seconds=float(np.percentile(row["inference_latencies_seconds"],95)),
            dropped=row["dropped"],collision=row["collision"],ever_bilateral=row["ever_bilateral"],ever_lifted=row["ever_lifted"],reason=row["reason"],diagnosis=diagnosis))
    with (destination/"逐条测试结果.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    write(destination/"failure-diagnoses.json",failure_counts)
    write(destination/"video-index.json",evidence)
    write(destination/"inference-latencies.json",latencies)
    m=complete["metrics"]
    comparisons=[]
    for light in ["normal","dim"]:
        base={r["index"]:r for r in rows if r["model_condition"]=="base" and r["lighting"]==light}
        fine={r["index"]:r for r in rows if r["model_condition"]=="finetuned27000" and r["lighting"]==light}
        comparisons.append(dict(lighting=light,positions=len(base),both_succeeded=sum(base[i]["success"] and fine[i]["success"] for i in base),
          only_finetuned_succeeded=sum(not base[i]["success"] and fine[i]["success"] for i in base),
          only_base_succeeded=sum(base[i]["success"] and not fine[i]["success"] for i in base),
          neither_succeeded=sum(not base[i]["success"] and not fine[i]["success"] for i in base)))
    write(destination/"paired-comparison.json",comparisons)
    light_comparison=[]
    for model in ["base","finetuned27000"]:
        normal={r["index"]:r for r in rows if r["model_condition"]==model and r["lighting"]=="normal"}
        dim={r["index"]:r for r in rows if r["model_condition"]==model and r["lighting"]=="dim"}
        light_comparison.append(dict(model=model,same_seed_count=len(dim),
              normal_successes=sum(normal[i]["success"] for i in dim),dim_successes=sum(dim[i]["success"] for i in dim),
              both_succeeded=sum(normal[i]["success"] and dim[i]["success"] for i in dim),
              only_normal_succeeded=sum(normal[i]["success"] and not dim[i]["success"] for i in dim),
              only_dim_succeeded=sum(not normal[i]["success"] and dim[i]["success"] for i in dim),
              neither_succeeded=sum(not normal[i]["success"] and not dim[i]["success"] for i in dim)))
    write(destination/"paired-lighting-comparison.json",light_comparison)
    light_table="\n".join(f"| {x['model']} | {x['normal_successes']}/10 | {x['dim_successes']}/10 |" for x in light_comparison)
    number=lambda x:"—" if x is None else f"{x:.3f}"
    lines=[]
    for key in ["base-normal","finetuned27000-normal","base-dim","finetuned27000-dim"]:
        x=m[key];label=("Base" if key.startswith("base") else "微调27,000步")+(" / 正常灯光" if key.endswith("normal") else " / 灯光90%")
        lines.append(f"| {label} | {x['successes']}/{x['attempts']} | {x['success_rate']:.1%} | {number(x['successful_sim_time_mean_s'])} | {x['mean_wall_time_all_trials_s']:.2f} | {x['inference_call_mean_s']*1000:.1f} | {x['inference_call_p95_s']*1000:.1f} | {x['drop_rate']:.1%} | {x['collision_rate']:.1%} |")
    subgroups=[]
    for model in ["base","finetuned27000"]:
        for category in ["interior","boundary"]:
            x=m[model+"-normal-"+category]
            subgroups.append(f"| {model} | {category} | {x['successes']}/{x['attempts']} | {x['success_rate']:.1%} |")
    diagnostic="\n".join(f"- {key}："+"；".join(f"{category}{count}次" for category,count in counts.items()) for key,counts in failure_counts.items())
    text=f"""# 第八步：SmolVLA固定测试集对照评估

已完成50个独立新种子上的Base与27,000步微调模型配对测试。正常灯光每种模型50条轨迹，其中40个内部位置、10个边界位置。另对固定10个种子各追加一次轻微降光，共120条轨迹。降光复测使用相同10个位置，不计作新增独立位置。

第六步开发验证为4/5，选定权重后才冻结本测试集。历史登记位置共排除{read(source/'position-exclusion-audit.json')['excluded_unique_xy']}个；新位置与历史位置最小距离为{read(source/'position-exclusion-audit.json')['minimum_distance_to_registered_old_positions_m']*1000:.3f}毫米。测试不是远离训练分布的任务泛化，范围沿用原任务工作区。

| 模型与条件 | 成功次数 | 成功率 | 成功任务平均仿真秒 | 所有轨迹平均墙钟秒 | 推理均值ms | 推理P95 ms | 掉落率 | 碰撞率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

只有完整释放、充分张开与撤离后，方块在托盘内独立稳定2秒，并在含1秒初始静置的30仿真秒内完成，才计成功。失败样本不进入“成功任务平均仿真秒”；无成功时为“—”。所有轨迹的仿真耗时、墙钟耗时及原生失败原因见CSV和JSON。

| 正常灯光模型 | 位置类别 | 成功次数 | 成功率 |
|---|---|---:|---:|
{chr(10).join(subgroups)}

光照影响另按同一组10个种子配对比较，避免把正常灯光50个位置与降光10个位置的分布差异当作光照效应：

| 模型 | 相同10种子正常灯光成功次数 | 降光成功次数 |
|---|---:|---:|
{light_table}

这组配对样本量为10。降光改变了闭环动作与结果，不能据此断言降光通常会改善或损害模型表现。逐种子转换次数见`paired-lighting-comparison.json`。

Base来自原官方SmolVLA的既有8维任务接口适配模型，optimizer更新为0，官方权重严格一致证明保留在原初始化目录。两组使用相同的最终图像裁剪、Train80归一化统计、动作语义、计算精度、50Hz插值控制、1秒重规划、夹爪映射及全部物理保护。对照衡量固定任务接口下权重微调的效果，未将Base视为已学习本任务的机器人控制器。

光照测试把MuJoCo灯光的ambient、diffuse和specular强度乘0.9，未对输入像素再做亮度变换。模型仅接收RGB、8维实测机器人状态与统一任务指令，仿真物体真值仅用于复位、评分和事后诊断。边界与降光样本量有限，报告给出实际次数，不据此断言总体成功率。

推理延迟包含预处理、模型、反归一化及CUDA同步，剔除每次加载后的2次预热和模型加载时间。仿真在推理期间暂停，仿真30秒与实际墙钟时间不同；当前结果不能作为真实时间控制性能证明。

掉落包括方块掉出台面或原生搬运中丢失载荷的终止，正常放置释放不计掉落。碰撞包含原生机器人与障碍接触或碰撞终止。接触力等其他保护终止在原始reason单独保留，未把所有保护失败合并成碰撞。

失败分类仅根据已保存的评分日志作事后诊断，不能仅凭这些分类断言具体网络内部原因：

{diagnostic}

`videos/`保存全部120条视频、末帧、结果和协议，包含全部失败。`video-index.json`将每条视频与失败分类及WSL中的完整命令和动作块对应。`逐条测试结果.csv`供作业统计；`complete.json`保留精确分组指标。模型与执行参数在整个测试中保持冻结，没有依据测试结果重新选模。

微调权重SHA256：`{protocol['weights_sha256']['finetuned27000']}`。

Base权重SHA256：`{protocol['weights_sha256']['base']}`。

固定协议SHA256：`{sha(source/'protocol.json')}`。
"""
    (destination/"第八步测试报告.md").write_text(text,encoding="utf-8")
    write(destination/"delivery-verification.json",dict(passed=True,trial_count=120,all_expected_pairs_present=True,
          models_unchanged=True,external_assets_unchanged=True,protocol_unchanged_since_launch=True,
          recomputed_metrics_exactly_match=True,all_success_terminal_evidence_checked=True,
          all_videos_exported=True))
    print(dict(exported=destination,normal_finetuned_successes=m["finetuned27000-normal"]["successes"],normal_base_successes=m["base-normal"]["successes"]),flush=True)
