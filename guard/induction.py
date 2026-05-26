import time, csv
import torch
import numpy as np
import pandas as pd
import guard.world as world
from guard.world import World, Agent, Blade, Boundary, device

dt = 0.04
period_length = 2
period_count = 3
move_length = period_length / 2  
ring_size = 15.0
action_count = 9
steps_per_move = round(move_length / dt)
move_count = 2 * period_count
death0_value = 1e6          

def load_initial(csv_path):
    df = pd.read_csv(csv_path)
    r = df.iloc[0]
    state = {
        'a0': (r.a0_x, r.a0_y, r.a0_vx, r.a0_vy),
        'b0': (r.b0_x, r.b0_y, r.b0_vx, r.b0_vy),
        'a1': (r.a1_x, r.a1_y, r.a1_vx, r.a1_vy),
        'b1': (r.b1_x, r.b1_y, r.b1_vx, r.b1_vy),
    }
    corners = df[['c0x','c0y','c1x','c1y','c2x','c2y','c3x','c3y']].iloc[0].to_numpy().reshape(4,2)
    return state, corners

def build_sim(batch, init_state, corners):
    w = World(batch, dt)
    a0 = Agent(w, 1); b0 = Blade(w, a0)
    a1 = Agent(w, 2); b1 = Blade(w, a1)
    w.boundary = Boundary(w)
    cor = torch.tensor(corners,dtype=world.physics_dtype).unsqueeze(0).expand(batch,4,2).contiguous()
    w.boundary.setup(cor)
    def configure_circle(circle, key):
        x,y,vx,vy = init_state[key]
        circle.position = torch.tensor([[x,y]],dtype=world.physics_dtype).expand(batch,2).contiguous()
        circle.velocity = torch.tensor([[vx,vy]],dtype=world.physics_dtype).expand(batch,2).contiguous()
    configure_circle(a0,'a0'); configure_circle(b0,'b0'); configure_circle(a1,'a1'); configure_circle(b1,'b1')
    w.complete = torch.zeros(batch,1).bool()
    w.time = 0.0
    return w, a0, b0, a1, b1

def recover_equilibrium_path(leaf):
    cur = leaf.reshape(*([action_count] * move_count))
    chosen = []
    for h in range(move_count):
        cand_vals = []
        for a in range(action_count):
            sub = cur[a]
            for hh in range(move_count - 1, h, -1):
                axis = hh - h - 1
                sub = sub.amin(dim=axis) if (hh % 2 == 0) else sub.amax(dim=axis)
            cand_vals.append(sub.item())
        cand_vals = np.array(cand_vals)
        a_star = int(np.argmin(cand_vals)) if (h % 2 == 0) else int(np.argmax(cand_vals))
        chosen.append(a_star)
        cur = cur[a_star]
    return chosen
 
def export_equilibrium_path(path_actions, init_state, corners, out_csv):
    s, a0, b0, a1, b1 = build_sim(1, init_state, corners)
    f = open(out_csv, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        "frame","time","life0","life1",
        "a0_x","a0_y","a0_vx","a0_vy",
        "b0_x","b0_y","b0_vx","b0_vy",
        "a1_x","a1_y","a1_vx","a1_vy",
        "b1_x","b1_y","b1_vx","b1_vy",
        "grad_a0_vx","grad_a0_vy","grad_a1_vx","grad_a1_vy",
        "reward","value_estimate","action0","action1",
        'c0x','c0y','c1x','c1y','c2x','c2y','c3x','c3y'
    ])
    cur0, cur1 = 0, 0
    frame = 0
    a0_dead = a1_dead = False
    for h in range(move_count):
        if h % 2 == 0: cur0 = path_actions[h]
        else:          cur1 = path_actions[h]
        a0.action = torch.tensor([cur0], dtype=torch.int, device=device)
        a1.action = torch.tensor([cur1], dtype=torch.int, device=device)
        for _ in range(steps_per_move):
            s.step()
            frame += 1
            gap0 = torch.norm(a0.position - b1.position, dim=1)
            gap1 = torch.norm(a1.position - b0.position, dim=1)
            a0_dead = a0_dead or bool(gap0.item() < ring_size)
            a1_dead = a1_dead or bool(gap1.item() < ring_size)
            life0 = 0 if a0_dead else 1
            life1 = 0 if a1_dead else 1
            def xy(c): return [c.position[0,0].item(), c.position[0,1].item(),
                               c.velocity[0,0].item(), c.velocity[0,1].item()]
            row = [frame, s.time, life0, life1]
            row += xy(a0); row += xy(b0); row += xy(a1); row += xy(b1)
            row += [0,0,0,0, 0, 0, cur0, cur1]   # no costate/reward/value in equilibrium replay
            if frame == 1:
                row += [float(v) for v in corners.reshape(-1)]
            w.writerow(row)
    f.close()
    print(f'wrote equilibrium path to {out_csv} (a0_dead={a0_dead}, a1_dead={a1_dead})')

def main():
    init_state, corners = load_initial('simulation/simulation.csv')
    total = action_count ** move_count
    t0 = time.perf_counter()
    grids = torch.cartesian_prod(*[torch.arange(action_count) for _ in range(move_count)])
    if grids.dim()==1: grids = grids.unsqueeze(1)
    grids = grids.to(device)
    s, a0, b0, a1, b1 = build_sim(total, init_state, corners)
    charge = torch.zeros(total, device=device)
    maxcharge = torch.zeros(total, device=device)
    a0_dead = torch.zeros(total, dtype=torch.bool, device=device)
    a1_dead = torch.zeros(total, dtype=torch.bool, device=device)
    for h in range(move_count):
        actor = a0 if (h % 2 == 0) else a1
        actor.action = grids[:, h].to(torch.int)
        for _ in range(steps_per_move):
            s.step()
            gap0 = torch.norm(a0.position - b1.position, dim=1)
            gap1 = torch.norm(a1.position - b0.position, dim=1)
            a0_dead |= (gap0 < ring_size)
            a1_dead |= (gap1 < ring_size)
            alive = ~(a0_dead | a1_dead)
            a1d = torch.norm(a1.position, dim=1)
            inside = (a1d < ring_size) & alive
            charge = torch.clamp(charge + torch.where(inside, dt, -dt), min=0.0)
            maxcharge = torch.maximum(maxcharge, charge)
    leaf = torch.where(a0_dead, torch.full_like(maxcharge, death0_value), maxcharge)
    vals = leaf.reshape(*([action_count]*move_count))
    for h in range(move_count-1, -1, -1):
        vals = vals.amin(dim=h) if (h % 2 == 0) else vals.amax(dim=h)
    induction_value = vals.item()
    full = leaf.reshape(*([action_count]*move_count))
    first_move_vals = []
    for a in range(action_count):
        sub = full[a]
        for h in range(move_count-1, 0, -1):
            axis = h-1
            sub = sub.amin(dim=axis) if (h % 2 == 0) else sub.amax(dim=axis)
        first_move_vals.append(sub.item())
    first_move_vals = np.array(first_move_vals)
    best_a0 = int(np.argmin(first_move_vals))
    eq_path = recover_equilibrium_path(leaf)
    elapsed = time.perf_counter() - t0
    df = pd.read_csv('simulation/simulation.csv')
    horizon_t = move_count * move_length
    sub = df[df.time <= horizon_t]
    print('  first-move values (induction value if agent0 picks each, then optimal play):')
    for a in range(action_count):
        ang = 'null' if a == 0 else f'{(a-1)*45}deg'
        marker = '  <- optimal' if a == best_a0 else ''
        print(f'    action {a} ({ang:>6}): {first_move_vals[a]:.3f}{marker}')
    print(f'compute time: {elapsed:.2f}s')
    print(f'horizon: {horizon_t:.1f}s')
    print(f'induction value: {induction_value:.3f}')
    export_equilibrium_path(eq_path, init_state, corners, 'simulation/induction.csv')

if __name__ == '__main__':
    main()

