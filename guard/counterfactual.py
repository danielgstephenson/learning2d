"""
Counterfactual: from t=10, agent0 replays its ORIGINAL recorded actions while
agent1 is frozen to the null action (0) for the remainder.

Purpose: test whether the inward spiral / mutual-circling stalemate is being
DRIVEN by agent1's active circling. If freezing agent1 dissolves the spiral
(agent0's blade reaches the ring, co-rotation breaks), the stalemate was
sustained by agent1's active blade1 control. If the spiral persists, it is
intrinsic to agent0's own motion + spring dynamics.

This is a single batch=1 forward rollout -- no game tree, no optimization.
Physics is the real engine from physics.py. Output is logged in game.py CSV
format so analysis.r can plot it.
"""
import csv
import torch
import numpy as np
import pandas as pd
import physics as physics
from physics import World, Agent, Blade, Boundary, device

dt = 0.04
ring_size = 15.0
branch_time = 15.0
csv_path = 'simulation/simulation.csv'
out_csv  = 'simulation/counterfactual.csv'

def build_sim(init_state, corners):
    w = World(1, dt)
    a0 = Agent(w, 1); b0 = Blade(w, a0)
    a1 = Agent(w, 2); b1 = Blade(w, a1)
    w.boundary = Boundary(w)
    cor = torch.tensor(corners, dtype=physics.physics_dtype).unsqueeze(0).expand(1,4,2).contiguous()
    w.boundary.setup(cor)
    def cfg(circle, key):
        x,y,vx,vy = init_state[key]
        circle.position = torch.tensor([[x,y]], dtype=physics.physics_dtype).contiguous()
        circle.velocity = torch.tensor([[vx,vy]], dtype=physics.physics_dtype).contiguous()
    cfg(a0,'a0'); cfg(b0,'b0'); cfg(a1,'a1'); cfg(b1,'b1')
    w.complete = torch.zeros(1,1).bool()
    w.time = 0.0
    return w, a0, b0, a1, b1

def main():
    df = pd.read_csv(csv_path)
    frame0 = int((df.time - branch_time).abs().idxmin())
    t0 = float(df.time.iloc[frame0])
    r = df.iloc[frame0]
    init_state = {
        'a0': (r.a0_x, r.a0_y, r.a0_vx, r.a0_vy),
        'b0': (r.b0_x, r.b0_y, r.b0_vx, r.b0_vy),
        'a1': (r.a1_x, r.a1_y, r.a1_vx, r.a1_vy),
        'b1': (r.b1_x, r.b1_y, r.b1_vx, r.b1_vy),
    }
    corners = df[['c0x','c0y','c1x','c1y','c2x','c2y','c3x','c3y']].iloc[0].to_numpy().reshape(4,2)
    # agent0's recorded actions from the branch frame onward
    a0_actions = df['action0'].iloc[frame0:].to_numpy().astype(int)

    s, a0, b0, a1, b1 = build_sim(init_state, corners)

    f = open(out_csv, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        "frame","time","life0","life1",
        "a0_x","a0_y","a0_vx","a0_vy","b0_x","b0_y","b0_vx","b0_vy",
        "a1_x","a1_y","a1_vx","a1_vy","b1_x","b1_y","b1_vx","b1_vy",
        "grad_a0_vx","grad_a0_vy","grad_a1_vx","grad_a1_vy",
        "reward","value_estimate","action0","action1",
        'c0x','c0y','c1x','c1y','c2x','c2y','c3x','c3y'
    ])

    a0_dead = a1_dead = False
    charge = 0.0; maxc = 0.0
    n = len(a0_actions)
    for k in range(n):
        act0 = int(a0_actions[k])
        a0.action = torch.tensor([act0], dtype=torch.int, device=device)
        a1.action = torch.tensor([0],    dtype=torch.int, device=device)  # FROZEN null
        s.step()
        gap0 = torch.norm(a0.position - b1.position, dim=1)
        gap1 = torch.norm(a1.position - b0.position, dim=1)
        a0_dead = a0_dead or bool(gap0.item() < ring_size)
        a1_dead = a1_dead or bool(gap1.item() < ring_size)
        a1d = torch.norm(a1.position, dim=1).item()
        alive = not (a0_dead or a1_dead)
        charge = max(0.0, charge + (dt if (a1d < ring_size and alive) else -dt))
        maxc = max(maxc, charge)
        def xy(c): return [c.position[0,0].item(), c.position[0,1].item(),
                           c.velocity[0,0].item(), c.velocity[0,1].item()]
        row = [k+1, t0 + s.time, 0 if a0_dead else 1, 0 if a1_dead else 1]
        row += xy(a0); row += xy(b0); row += xy(a1); row += xy(b1)
        row += [0,0,0,0, 0,0, act0, 0]
        if k == 0:
            row += [float(v) for v in corners.reshape(-1)]
        w.writerow(row)
    f.close()
    print(f'branched at t={t0:.2f} (frame {frame0}); replayed {n} frames of agent0 actions with agent1 frozen null')
    print(f'max charge in counterfactual: {maxc:.2f}s   (a0_dead={a0_dead}, a1_dead={a1_dead})')
    print(f'wrote {out_csv}')

if __name__ == '__main__':
    main()