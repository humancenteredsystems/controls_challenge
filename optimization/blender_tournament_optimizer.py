#!/usr/bin/env python3
"""
Blender Tournament Optimizer - Neural Network Architecture Search for PID Blending
Enhanced terminal reporting with banners, parameters logging, progress bars, and summaries.
"""

import sys
import os
import argparse
import json
import random
import numpy as np
import hashlib
from pathlib import Path
import logging

# Add parent directory to path to find tinyphysics
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tinyphysics_custom import run_rollout, TinyPhysicsModel
from controllers.shared_pid import SpecializedPID
from utils.logging import print_banner, print_params, print_summary, tqdm, EMOJI_PARTY, EMOJI_TROPHY, EMOJI_OK

def cleanup_artifacts() -> None:
    """Remove leftover temporary controllers and blender models."""
    base_dir = Path(__file__).parent.parent
    for path in (base_dir / "controllers").glob("temp_*.py"):
        try: path.unlink()
        except: pass
    for path in (base_dir / "models").glob("blender_*.onnx"):
        try: path.unlink()
        except: pass

def create_training_data_from_archive(archive_path, data_files, model, num_samples=5000):
    """Generate training data from PID tournament archive (with progress bar)."""
    print(f"\n🔬 Generating training data from archive: {archive_path}")
    with open(archive_path, 'r') as f:
        archive = json.load(f)
    entries = archive.get('archive', [])[:20]
    valid = [e for e in entries if isinstance(e.get('stats'), dict) and 'avg_total_cost' in e['stats']]
    top = sorted(valid or entries, key=lambda x: x['stats'].get('avg_total_cost', np.inf))[:10]
    splits = num_samples // len(top)
    samples = []
    for idx, combo in enumerate(tqdm(top, desc="Combos", unit="combo"), start=1):
        combo_samples = []
        for _ in tqdm(range(splits), desc=" Samples", leave=False, unit="sample"):
            data_file = random.choice(data_files)
            best_blend = find_optimal_blend_weight(combo['low_gains'],
                                                  combo['high_gains'], data_file, model)
            state, error, future = extract_state_from_file(data_file)
            features = [
                state.v_ego, state.roll_lataccel, state.a_ego,
                error, state.error_integral, state.error_derivative,
                np.mean(future.lataccel), np.std(future.lataccel)
            ]
            combo_samples.append((features, best_blend))
        samples.extend(combo_samples)
        print(f"  ✅ Generated {len(combo_samples)} samples for combo {idx}/{len(top)}")
    while len(samples) < num_samples:
        combo = random.choice(top)
        data_file = random.choice(data_files)
        best_blend = find_optimal_blend_weight(combo['low_gains'], combo['high_gains'], data_file, model)
        state, error, future = extract_state_from_file(data_file)
        features = [
            state.v_ego, state.roll_lataccel, state.a_ego,
            error, state.error_integral, state.error_derivative,
            np.mean(future.lataccel), np.std(future.lataccel)
        ]
        samples.append((features, best_blend))
    print_summary("Pre-Training Data Summary", {
        "total_samples": len(samples),
        "samples_per_combo": splits,
        "combos": len(top)
    })
    out = Path("plans/blender_training_data.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(samples, f, indent=2)
    return samples

def find_optimal_blend_weight(pid1_params, pid2_params, data_file, model):
    """Find optimal blend weight via discrete search using run_rollout."""
    from optimization import generate_blended_controller
    best_cost, best_blend = float('inf'), 0.5
    for blend in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        content = generate_blended_controller(pid1_params, pid2_params)
        content = content.replace('if v_ego < 40:', f'if False:').replace(
            'weights = [0.8, 0.2]', f'weights = [{blend}, {1-blend}]').replace(
            'weights = [0.2, 0.8]', f'weights = [{blend}, {1-blend}]')
        name = f"temp_blend_{hashlib.md5(str(blend).encode()).hexdigest()[:8]}"
        path = Path("controllers") / f"{name}.py"
        try:
            with open(path, 'w') as f:
                f.write(content)
            cost, _, _ = run_rollout(data_file, name, model)
            if cost["total_cost"] < best_cost:
                best_cost, best_blend = cost["total_cost"], blend
        except Exception:
            pass
        finally:
            if path.exists():
                path.unlink()
    return best_blend

def extract_state_from_file(data_file):
    """Extract random state and future plan for blending."""
    import pandas as pd
    df = pd.read_csv(data_file)
    idx = random.randint(100, min(len(df)-50, 400))
    from collections import namedtuple
    State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego', 'error_integral', 'error_derivative'])
    FuturePlan = namedtuple('FuturePlan', ['lataccel'])
    ACC_G = 9.81
    roll = np.sin(df.iloc[idx]['roll']) * ACC_G
    v = df.iloc[idx]['vEgo']
    a = df.iloc[idx]['aEgo']
    error = random.uniform(-1, 1)
    state = State(roll, v, a, error*0.1, error*0.05)
    future = FuturePlan(lataccel=df['targetLateralAcceleration'].iloc[idx:idx+20].tolist())
    return state, error, future

def create_random_blender_architecture():
    """Create random BlenderNet architecture parameters."""
    hidden = []
    for _ in range(random.randint(2,3)):
        hidden.append(random.choice([16,24,32,48]))
    dropout = random.choice([0.05,0.1,0.15])
    return {'hidden_sizes': hidden, 'dropout_rate': dropout,
            'id': hashlib.md5(str(hidden+[dropout]).encode()).hexdigest()[:8]}

def train_blender_architecture(architecture, training_data, epochs=100):
    """Train BlenderNet and export to ONNX."""
    from neural_blender_net import BlenderNet, train_blender_net
    mid = architecture['id']
    path = Path("models") / f"blender_{mid}.onnx"
    path.parent.mkdir(exist_ok=True)
    print(f"    Training architecture {mid} for {epochs} epochs...")
    model = train_blender_net(training_data, epochs=epochs)
    model.export_to_onnx(str(path))
    return str(path)

def evaluate_blender_architecture(architecture, training_data, data_files, model, baseline, max_files=20):
    """Evaluate architecture performance vs baseline."""
    onnx = train_blender_architecture(architecture, training_data)
    pid_pairs = get_top_pid_pairs_from_archive()
    costs = []
    files = random.sample(data_files, min(max_files, len(data_files)))

    try:
        for f in files:
            for low, high in pid_pairs[:3]:
                mod = _make_temp_neural_controller(low, high, onnx, architecture['id'])
                try:
                    c, _, _ = run_rollout(f, mod, model)
                    costs.append(c["total_cost"])
                except Exception:
                    costs.append(1e3)
                finally:
                    cleanup_temp_controller(mod)
        return float(np.mean(costs)) if costs else float('inf')
    finally:
        try:
            Path(onnx).unlink()
        except FileNotFoundError:
            pass

def tournament_selection_and_mutation(pop, elite_pct=0.3, mutation_rate=0.2):
    """Select elites and generate mutated offspring."""
    pop.sort(key=lambda x:x['cost'])
    elites=int(len(pop)*elite_pct)
    new=pop[:elites]
    while len(new)<len(pop):
        p=random.choice(pop[:len(pop)//2])
        child=p.copy()
        if random.random()<mutation_rate:
            i=random.randrange(len(child['hidden_sizes']))
            child['hidden_sizes'][i]=random.choice([16,24,32,48])
        if random.random()<mutation_rate:
            child['dropout_rate']=random.choice([0.05,0.1,0.15])
        child['id']=hashlib.md5(str(child['hidden_sizes']+[child['dropout_rate']]).encode()).hexdigest()[:8]
        child['cost']=float('inf')
        new.append(child)
    return new

def get_top_pid_pairs_from_archive(archive_path="plans/tournament_archive.json"):
    """Get top PID pairs for evaluation."""
    with open(archive_path,'r') as f:
        arc=json.load(f)
    perf=[p for p in arc.get('archive',[]) if 'stats' in p and 'avg_total_cost' in p['stats']]
    top=sorted(perf,key=lambda x:x['stats']['avg_total_cost'])[:5]
    return [(c['low_gains'],c['high_gains']) for c in top]

def get_tournament_2_baseline(archive_path="plans/tournament_archive.json"):
    """Get best cost from Tournament #2."""
    try:
        with open(archive_path,'r') as f:
            arc=json.load(f)
        vals=[p['stats']['avg_total_cost'] for p in arc.get('archive',[]) if 'avg_total_cost' in p['stats']]
        b=min(vals)
        print(f"📊 Tournament #2 baseline to beat: {b:.2f}")
        return b
    except:
        print("⚠️ Could not load baseline, using 100.0")
        return 100.0

def _make_temp_neural_controller(low,high,onnx,arch_id):
    """Write temp neural controller file."""
    from optimization import generate_neural_blended_controller
    code=generate_neural_blended_controller(low,high,onnx)
    name=f"temp_neural_{hashlib.md5((str(arch_id)+onnx).encode()).hexdigest()[:8]}"
    path=Path("controllers")/f"{name}.py"
    Path("controllers").mkdir(exist_ok=True)
    with open(path,"w") as f: f.write(code)
    return name

def cleanup_temp_controller(name):
    """Remove temp controller file."""
    p=Path("controllers")/f"{name}.py"
    if p.exists(): p.unlink()

def create_champion_controller(best_arch, best_entry, cost, archive_path):
    """Create final champion controller."""
    print(f"{EMOJI_TROPHY} Creating champion controller...")
    # copy champion ONNX
    champ_path="models/neural_blender_champion.onnx"
    # existing ONNX from best_arch assumed at models/blender_<id>.onnx
    src=f"models/blender_{best_arch['id']}.onnx"
    if Path(src).exists(): Path(src).rename(champ_path)
    pid1, pid2 = best_entry['low_gains'], best_entry['high_gains']
    from optimization import generate_neural_blended_controller
    content=generate_neural_blended_controller(pid1,pid2,champ_path)
    ctrl_path="controllers/neural_blended_champion.py"
    with open(ctrl_path,"w") as f: f.write(content)
    print(f"{EMOJI_OK} Champion controller ready: {ctrl_path}")

def run_blender_tournament(archive_path, data_files, model_path,
                           rounds=15, pop_size=20, max_files=20, samples_per_combo=50):
    """Run tournament optimization for BlenderNet architectures."""
    print_banner(5, "Neural Blender Tournament (Architecture Search)")
    print_params({
        "archive": archive_path,
        "rounds": rounds,
        "population": pop_size,
        "max_files": max_files,
        "samples_per_combo": samples_per_combo
    })
    baseline=get_tournament_2_baseline(archive_path)
    model=TinyPhysicsModel(model_path,debug=False)
    print("Blender Tournament: GPU Enabled")
    data=create_training_data_from_archive(archive_path,data_files,model,samples_per_combo)
    pop=[dict(create_random_blender_architecture(), cost=float('inf')) for _ in range(pop_size)]
    best={'cost':float('inf')}
    for r in range(1,rounds+1):
        print(f"\n🏆 Round {r}/{rounds}")
        for arch in tqdm(pop,desc="Evaluating architectures",unit="arch"):
            if arch['cost']==float('inf'):
                c=evaluate_blender_architecture(arch,data,data_files,model,baseline,max_files)
                arch['cost']=c
                if c<best['cost']:
                    best=arch.copy()
                    print(f"🎉 New overall best: {c:.2f} (id:{arch['id']})")
        rb=min(pop,key=lambda x:x['cost'])
        print(f"Round {r} complete — round best: {rb['cost']:.2f}")
        if r<rounds:
            pop=tournament_selection_and_mutation(pop)
            print("➡️ Next generation created")
    imp=baseline-best['cost']
    print("\n🎯 Stage 5 Results:")
    print(f"  Baseline: {baseline:.2f}")
    print(f"  Best:     {best['cost']:.2f}")
    print(f"  Improvement: {imp:.2f}")
    print(f"  Hyperparameters: {best['hidden_sizes']}, dropout={best['dropout_rate']}")
    # champion PID entry
    with open(archive_path,'r') as f: arc=json.load(f)
    entries=arc.get('archive',[])
    champ_entry=min(entries, key=lambda x:x['stats'].get('avg_total_cost',float('inf')))
    create_champion_controller(best, champ_entry, best['cost'], archive_path)
    return best

def main():
    cleanup_artifacts()
    p=argparse.ArgumentParser(description='Blender Tournament Optimizer')
    p.add_argument('--archive',default='plans/tournament_archive.json')
    p.add_argument('--rounds',type=int,default=15)
    p.add_argument('--pop_size',type=int,default=20)
    p.add_argument('--max_files',type=int,default=20)
    p.add_argument('--model_path',default='models/tinyphysics.onnx')
    p.add_argument('--data_seed',type=int,default=None)
    p.add_argument('--samples_per_combo',type=int,default=50)
    args=p.parse_args()
    if args.data_seed is not None:
        random.seed(args.data_seed); np.random.seed(args.data_seed)
    files=[str(f) for f in Path("data").glob("*.csv")]
    if not files:
        print("❌ No data files found"); sys.exit(1)
    random.shuffle(files)
    print(f"Found {len(files)} data files")
    run_blender_tournament(args.archive,files,args.model_path,
                           args.rounds,args.pop_size,args.max_files,args.samples_per_combo)
    return 0

if __name__=="__main__":
    sys.exit(main())
