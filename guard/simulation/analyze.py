import csv
import math
import sys
import os

KILL_THRESHOLD = 15     # gap between agent and enemy blade that causes death
RING_THRESHOLD = 8      # center_dist for agent1 to be charging (ringSize - agent1.radius)
CHARGE_TARGET  = 1.0    # normalized charge needed to win

def parse_life(val):
    if val in ('True', 'true'): return 1
    if val in ('False', 'false'): return 0
    return int(float(val))

def dist_from_origin(x, y):
    return math.sqrt(float(x)**2 + float(y)**2)

def dist_between(ax, ay, bx, by):
    return math.sqrt((float(ax) - float(bx))**2 + (float(ay) - float(by))**2)

def load(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def find_terminal_index(rows):
    for i, r in enumerate(rows):
        if float(r['charge']) >= CHARGE_TARGET:
            return i
        if parse_life(r['life1']) == 0:
            return i
    return len(rows) - 1

def determine_outcome(rows):
    last = rows[-1]
    if float(last['charge']) >= CHARGE_TARGET:
        return "Agent1 wins  —  fully charged the ring"
    if parse_life(last['life1']) == 0:
        return "Agent0 wins  —  killed agent1"
    return f"No winner  —  simulation ended at t = {float(last['time']):.2f}s"

def find_charging_periods(rows):
    periods = []
    active = False
    period_rows = []
    for r in rows:
        if float(r['charge']) > 0:
            if not active:
                active = True
                period_rows = []
            period_rows.append(r)
        else:
            if active:
                active = False
                periods.append(period_rows)
                period_rows = []
    if active and period_rows:
        periods.append(period_rows)
    return periods

def load_horizon():
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'checkpoint.pt')
    try:
        import torch
        ck = torch.load(checkpoint_path, weights_only=False)
        return ck.get('horizon', None), ck.get('batch', None)
    except Exception:
        return None, None

def find_post_death(all_rows, terminal_idx):
    terminal = all_rows[terminal_idx]
    agent0_died = parse_life(terminal['life0']) == 0
    agent1_died = parse_life(terminal['life1']) == 0
    post = all_rows[terminal_idx + 1:]
    result = {}
    if agent1_died and not agent0_died:
        for r in post:
            if parse_life(r['life0']) == 0:
                result['agent0_death_time'] = float(r['time'])
                result['survival_margin'] = float(r['time']) - float(terminal['time'])
                break
    if agent0_died and not agent1_died:
        for r in post:
            if parse_life(r['life1']) == 0:
                result['agent1_death_time'] = float(r['time'])
                result['survival_margin'] = float(r['time']) - float(terminal['time'])
                break
    return result

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'simulation.csv')
    all_rows = load(path)
    terminal_idx = find_terminal_index(all_rows)
    rows = all_rows[:terminal_idx + 1]
    post_death = find_post_death(all_rows, terminal_idx)
    first, last = rows[0], rows[-1]
    horizon, batch = load_horizon()

    # ------------------------------------------------------------------ #
    print("=" * 62)
    print("  SIMULATION SUMMARY")
    print("=" * 62)
    print(f"  Frames   : {len(rows)}  (of {len(all_rows)} logged)")
    print(f"  Duration : {float(last['time']):.2f} seconds")
    print(f"  Outcome  : {determine_outcome(rows)}")
    if 'agent0_death_time' in post_death:
        print(f"  Agent0 survived {post_death['survival_margin']:.2f}s after game end (died at t = {post_death['agent0_death_time']:.2f}s)")
    elif 'agent1_death_time' in post_death:
        print(f"  Agent1 survived {post_death['survival_margin']:.2f}s after game end (died at t = {post_death['agent1_death_time']:.2f}s)")
    if horizon is not None:
        print(f"  Horizon  : {horizon}  ({horizon * 0.1:.1f}s planning), Batch: {batch}")
    value_est = float(first['value_estimate'])
    sign = "predicts agent1 wins" if value_est < 0.5 else "predicts agent0 wins"
    print(f"  Initial value estimate : {value_est:.3f}  ({sign})")

    # ------------------------------------------------------------------ #
    print()
    print("=" * 62)
    print("  STARTING CONDITIONS")
    print("=" * 62)
    a0_start = dist_from_origin(first['a0_x'], first['a0_y'])
    a1_start = dist_from_origin(first['a1_x'], first['a1_y'])
    print(f"  Agent0 dist from ring : {a0_start:.1f}  ({'near ring' if a0_start < 30 else 'far from ring'})")
    print(f"  Agent1 dist from ring : {a1_start:.1f}  ({'near ring' if a1_start < 30 else 'far from ring'})")
    blade0_a1 = dist_between(first['a1_x'], first['a1_y'], first['b0_x'], first['b0_y'])
    blade1_a0 = dist_between(first['a0_x'], first['a0_y'], first['b1_x'], first['b1_y'])
    print(f"  Blade0 -> agent1      : {blade0_a1:.1f}  (agent0 threat to agent1)")
    print(f"  Blade1 -> agent0      : {blade1_a0:.1f}  (agent1 threat to agent0)")

    # ------------------------------------------------------------------ #
    print()
    print("=" * 62)
    print("  TRAJECTORY  (sampled at ~12 points)")
    print("=" * 62)
    print(f"  {'Time':>6}  {'A0 dist':>7}  {'A1 dist':>7}  {'Blade0->A1':>10}  {'Charge':>6}  {'Reward':>8}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*6}  {'-'*8}")
    step = max(1, len(rows) // 12)
    indices = sorted(set(list(range(0, len(rows), step)) + [len(rows) - 1]))
    for i in indices:
        r = rows[i]
        t    = float(r['time'])
        a0d  = dist_from_origin(r['a0_x'], r['a0_y'])
        a1d  = dist_from_origin(r['a1_x'], r['a1_y'])
        b0a1 = dist_between(r['a1_x'], r['a1_y'], r['b0_x'], r['b0_y'])
        ch   = float(r['charge'])
        rw   = float(r['reward'])
        print(f"  {t:>6.2f}  {a0d:>7.1f}  {a1d:>7.1f}  {b0a1:>10.1f}  {ch:>6.2f}  {rw:>8.2f}")

    # ------------------------------------------------------------------ #
    print()
    print("=" * 62)
    print("  CHARGING PERIODS")
    print("=" * 62)
    periods = find_charging_periods(rows)
    if not periods:
        print("  Agent1 never entered the ring.")
    else:
        print(f"  Agent1 made {len(periods)} charging attempt(s).")
        print(f"  (Ring charges while agent1 center is within {RING_THRESHOLD} units of origin)")
        for i, period in enumerate(periods):
            t_start  = float(period[0]['time'])
            t_end    = float(period[-1]['time'])
            duration = t_end - t_start
            max_ch   = max(float(r['charge']) for r in period)
            pct      = 100 * max_ch / CHARGE_TARGET

            a0d_start = dist_from_origin(period[0]['a0_x'], period[0]['a0_y'])
            a0d_end   = dist_from_origin(period[-1]['a0_x'], period[-1]['a0_y'])
            direction = "closing in" if a0d_end < a0d_start else "moving away"

            min_b0_gap = min(dist_between(r['a1_x'], r['a1_y'], r['b0_x'], r['b0_y']) for r in period)
            complete   = max_ch >= CHARGE_TARGET

            print()
            print(f"  Attempt {i+1}:")
            print(f"    Time span   : t = {t_start:.2f}s  to  t = {t_end:.2f}s  ({duration:.2f}s long)")
            if complete:
                print(f"    Result      : CHARGE COMPLETE — agent1 wins")
            else:
                print(f"    Result      : Defended  (reached {max_ch:.2f} / {CHARGE_TARGET:.2f},  {pct:.0f}%)")
            print(f"    Agent0 dist : {a0d_start:.1f}  ->  {a0d_end:.1f}  ({direction})")
            print(f"    Closest blade0 got to agent1 : {min_b0_gap:.1f}  (kill threshold = {KILL_THRESHOLD})")

    # ------------------------------------------------------------------ #
    print()
    print("=" * 62)
    print("  DEFENSIVE SUMMARY")
    print("=" * 62)
    all_b0_gaps  = [dist_between(r['a1_x'], r['a1_y'], r['b0_x'], r['b0_y']) for r in rows]
    all_b1_gaps  = [dist_between(r['a0_x'], r['a0_y'], r['b1_x'], r['b1_y']) for r in rows]
    min_b0_gap   = min(all_b0_gaps)
    min_b0_time  = float(rows[all_b0_gaps.index(min_b0_gap)]['time'])
    min_b1_gap   = min(all_b1_gaps)
    min_b1_time  = float(rows[all_b1_gaps.index(min_b1_gap)]['time'])
    max_charge   = max(float(r['charge']) for r in rows)

    print(f"  Closest blade0 got to agent1 : {min_b0_gap:.1f} at t = {min_b0_time:.2f}s")
    print(f"    (kill threshold = {KILL_THRESHOLD}; {'KILL ACHIEVED' if min_b0_gap < KILL_THRESHOLD else 'no kill'})")
    print(f"  Closest blade1 got to agent0 : {min_b1_gap:.1f} at t = {min_b1_time:.2f}s")
    print(f"    (kill threshold = {KILL_THRESHOLD}; {'KILL ACHIEVED' if min_b1_gap < KILL_THRESHOLD else 'no kill'})")
    print(f"  Maximum charge reached       : {max_charge:.2f} / {CHARGE_TARGET:.2f}  ({100*max_charge/CHARGE_TARGET:.0f}%)")

    # ------------------------------------------------------------------ #
    print()
    print("=" * 62)
    print("  LATE GAME ANALYSIS  (final 20% of simulation)")
    print("=" * 62)
    late_rows = rows[int(len(rows) * 0.8):]
    t_start_late = float(late_rows[0]['time'])

    def grad_angle_deg(gx, gy):
        mag = math.sqrt(float(gx)**2 + float(gy)**2)
        if mag < 1e-6:
            return None, 0.0
        angle = math.degrees(math.atan2(float(gy), float(gx)))
        return angle, mag

    def ring_angle_deg(x, y):
        d = dist_from_origin(x, y)
        if d < 1e-6:
            return None
        return math.degrees(math.atan2(-float(y), -float(x)))  # toward origin

    def angle_diff(a, b):
        if a is None or b is None:
            return None
        d = (a - b + 180) % 360 - 180
        return d

    a0_speeds = [math.sqrt(float(r['a0_vx'])**2 + float(r['a0_vy'])**2) for r in late_rows]
    a1_speeds = [math.sqrt(float(r['a1_vx'])**2 + float(r['a1_vy'])**2) for r in late_rows]
    values    = [float(r['value_estimate']) for r in late_rows]
    actions0  = [int(r['action0']) for r in late_rows]
    actions1  = [int(r['action1']) for r in late_rows]

    print(f"  Period          : t = {t_start_late:.2f}s  to  t = {float(late_rows[-1]['time']):.2f}s")
    print(f"  Agent0 speed    : avg {sum(a0_speeds)/len(a0_speeds):.1f},  min {min(a0_speeds):.1f},  max {max(a0_speeds):.1f}")
    print(f"  Agent1 speed    : avg {sum(a1_speeds)/len(a1_speeds):.1f},  min {min(a1_speeds):.1f},  max {max(a1_speeds):.1f}")
    print(f"  Value estimate  : avg {sum(values)/len(values):.3f},  min {min(values):.3f},  max {max(values):.3f}")

    from collections import Counter
    a0_counts = Counter(actions0)
    a1_counts = Counter(actions1)
    top0 = a0_counts.most_common(3)
    top1 = a1_counts.most_common(3)
    print(f"  Agent0 actions  : {', '.join(f'action {a}={c}x ({100*c//len(actions0)}%)' for a,c in top0)}")
    print(f"  Agent1 actions  : {', '.join(f'action {a}={c}x ({100*c//len(actions1)}%)' for a,c in top1)}")

    print()
    print(f"  GRADIENT DIRECTIONS  (sampled at 8 points in late period)")
    print(f"  {'Time':>6}  {'A0 spd':>6}  {'A1 spd':>6}  {'G0 mag':>6}  {'G0->ring':>8}  {'G1 mag':>6}  {'G1->ring':>8}  {'Value':>7}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*7}")
    step = max(1, len(late_rows) // 8)
    sample_indices = sorted(set(list(range(0, len(late_rows), step)) + [len(late_rows) - 1]))
    for i in sample_indices:
        r = late_rows[i]
        t   = float(r['time'])
        sp0 = math.sqrt(float(r['a0_vx'])**2 + float(r['a0_vy'])**2)
        sp1 = math.sqrt(float(r['a1_vx'])**2 + float(r['a1_vy'])**2)
        g0_ang, g0_mag = grad_angle_deg(r['grad_a0_vx'], r['grad_a0_vy'])
        g1_ang, g1_mag = grad_angle_deg(r['grad_a1_vx'], r['grad_a1_vy'])
        r0_ang = ring_angle_deg(r['a0_x'], r['a0_y'])
        r1_ang = ring_angle_deg(r['a1_x'], r['a1_y'])
        diff0  = angle_diff(g0_ang, r0_ang)
        diff1  = angle_diff(g1_ang, r1_ang)
        val    = float(r['value_estimate'])
        def fmt_diff(d):
            if d is None: return '     n/a'
            label = 'toward' if abs(d) < 90 else 'away  '
            return f'{label} {abs(d):>3.0f}°'
        print(f"  {t:>6.2f}  {sp0:>6.1f}  {sp1:>6.1f}  {g0_mag:>6.3f}  {fmt_diff(diff0)}  {g1_mag:>6.3f}  {fmt_diff(diff1)}  {val:>7.3f}")
    print()

if __name__ == '__main__':
    main()
