import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from models import ValueModel
import world as world_module
from world import action_tensor
from torch.func import vmap, grad as fgrad

device = world_module.device
checkpoint_path = './checkpoints/checkpoint.pt'
value_model = ValueModel().eval()
checkpoint = torch.load(checkpoint_path, weights_only=False)
value_model.load_state_dict(checkpoint['value_model'])

# Read simulation CSV — set frame=-1 for last frame, or a specific frame index
frame = 300

rows = []
with open('./simulation/simulation.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
current = rows[frame - 1] if frame > 0 else rows[frame]

# Build reference state vector from final simulation row
# State order: a0vel(0:2), b0vel(2:4), a1vel(4:6), b1vel(6:8),
#              a0pos(8:10), b0pos(10:12), a1pos(12:14), b1pos(14:16),
#              charge(16), alive0(17), alive1(18), wallpoints(19:35)
ref_state = torch.zeros(1, 35, device=device)
# New state order: a0vel(0:2), a1vel(2:4), b0vel(4:6), b1vel(6:8),
#                  a0pos(8:10), a1pos(10:12), b0pos(12:14), b1pos(14:16)
ref_state[0, 0]  = float(current['a0vx'])
ref_state[0, 1]  = float(current['a0vy'])
ref_state[0, 2]  = float(current['a1vx'])
ref_state[0, 3]  = float(current['a1vy'])
ref_state[0, 4]  = float(current['b0vx'])
ref_state[0, 5]  = float(current['b0vy'])
ref_state[0, 6]  = float(current['b1vx'])
ref_state[0, 7]  = float(current['b1vy'])
# agent0 position: 8,9 (will be varied)
ref_state[0, 10] = float(current['a1x'])
ref_state[0, 11] = float(current['a1y'])
ref_state[0, 12] = float(current['b0x'])
ref_state[0, 13] = float(current['b0y'])
ref_state[0, 14] = float(current['b1x'])
ref_state[0, 15] = float(current['b1y'])
ref_state[0, 16] = 0.0  # charge
ref_state[0, 17] = float(current['life0'])
ref_state[0, 18] = float(current['life1'])

# Read wall points directly from simulation CSV
wp_keys = ['wp0x','wp0y','wp1x','wp1y','wp2x','wp2y','wp3x','wp3y',
           'wp4x','wp4y','wp5x','wp5y','wp6x','wp6y','wp7x','wp7y']
for i, key in enumerate(wp_keys):
    ref_state[0, 19 + i] = float(current[key])

# Compute fresh gradient from ref_state (unblurred, for comparison with CSV gradient)
gradient_fn = vmap(fgrad(lambda x: value_model(x).sum()))
fresh_grad = gradient_fn(ref_state)
vg0x_fresh = fresh_grad[0, 0].item()
vg0y_fresh = fresh_grad[0, 1].item()

N = 200
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
theta = np.linspace(0, 2*np.pi, 200)
ring_r = 13

# --- Plot 1: Value vs Agent0 position (velocity fixed at 0, centered on agent0) ---
lim_pos = 40
a0x_ref, a0y_ref = float(current['a0x']), float(current['a0y'])
xs = np.linspace(a0x_ref - lim_pos, a0x_ref + lim_pos, N)
ys = np.linspace(a0y_ref - lim_pos, a0y_ref + lim_pos, N)
XX, YY = np.meshgrid(xs, ys)
states = ref_state.repeat(N * N, 1)
positions = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=1),
                          dtype=torch.float32, device=device)
states[:, 8:10] = positions
with torch.no_grad():
    val_pos = torch.sigmoid(value_model(states)).cpu().numpy().reshape(N, N)

vmin, vmax = val_pos.min(), val_pos.max()
print(f'Position plot value range: [{vmin:.4f}, {vmax:.4f}]')
pos_df = np.column_stack([XX.ravel(), YY.ravel(), val_pos.ravel()])
np.savetxt('./simulation/value_pos.csv', pos_df, delimiter=',', header='x,y,value', comments='')
ax = axes[0]
im = ax.imshow(val_pos, extent=[a0x_ref-lim_pos, a0x_ref+lim_pos, a0y_ref-lim_pos, a0y_ref+lim_pos],
               origin='lower', cmap='RdYlGn', vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax, label='Value')
ax.plot(ring_r * np.cos(theta), ring_r * np.sin(theta), 'k-', lw=2, label='Ring boundary')
agent_r = 5
ax.plot(float(current['a0x']) + agent_r * np.cos(theta),
        float(current['a0y']) + agent_r * np.sin(theta),
        color='grey', lw=1.5, linestyle='--', label='Agent radius')
b1x, b1y = float(current['b1x']), float(current['b1y'])
ax.plot(b1x, b1y, 'b^', ms=8, label=f'Blade1')
ax.plot(float(current['a0x']), float(current['a0y']), 'bo', ms=8, markeredgecolor='k', label='Agent0 pos')
ax.annotate('', xy=(float(current['a0x']) + float(current['a0vx']),
                    float(current['a0y']) + float(current['a0vy'])),
            xytext=(float(current['a0x']), float(current['a0y'])),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.set_xlabel('Agent0 x'); ax.set_ylabel('Agent0 y')
ax.set_title(f'Value vs Position (frame {current["frame"]})')
ax.set_xlim(a0x_ref-lim_pos, a0x_ref+lim_pos)
ax.set_ylim(a0y_ref-lim_pos, a0y_ref+lim_pos)
ax.legend(); ax.set_aspect('equal')

# --- Plot 2: Value vs Agent0 velocity (position fixed at final value) ---
lim_vel = 1
a0vx_ref, a0vy_ref = float(current['a0vx']), float(current['a0vy'])
vs_x = np.linspace(a0vx_ref - lim_vel, a0vx_ref + lim_vel, N)
vs_y = np.linspace(a0vy_ref - lim_vel, a0vy_ref + lim_vel, N)
VX, VY = np.meshgrid(vs_x, vs_y)
states2 = ref_state.repeat(N * N, 1)
# Fix agent0 position at final simulation value
states2[:, 8] = float(current['a0x'])
states2[:, 9] = float(current['a0y'])
velocities = torch.tensor(np.stack([VX.ravel(), VY.ravel()], axis=1),
                           dtype=torch.float32, device=device)
states2[:, 0:2] = velocities
with torch.no_grad():
    val_vel = torch.sigmoid(value_model(states2)).cpu().numpy().reshape(N, N)

vmin2, vmax2 = val_vel.min(), val_vel.max()
print(f'Velocity plot value range: [{vmin2:.4f}, {vmax2:.4f}]')
vel_df = np.column_stack([VX.ravel(), VY.ravel(), val_vel.ravel()])
np.savetxt('./simulation/value_vel.csv', vel_df, delimiter=',', header='vx,vy,value', comments='')
ax2 = axes[1]
im2 = ax2.imshow(val_vel, extent=[a0vx_ref-lim_vel, a0vx_ref+lim_vel, a0vy_ref-lim_vel, a0vy_ref+lim_vel],
                 origin='lower', cmap='RdYlGn', vmin=vmin2, vmax=vmax2)
plt.colorbar(im2, ax=ax2, label='Value')
ax2.axhline(a0vy_ref, color='k', lw=0.5); ax2.axvline(a0vx_ref, color='k', lw=0.5)
ax2.plot(float(current['a0vx']), float(current['a0vy']), 'bo', ms=8, markeredgecolor='k', label='Agent0 vel')
vg0x, vg0y = float(current['vg0x']), float(current['vg0y'])
scale = lim_vel / max(abs(vg0x), abs(vg0y), 1e-9) * 0.3
ax2.annotate('', xy=(float(current['a0vx']) + vg0x * scale,
                     float(current['a0vy']) + vg0y * scale),
             xytext=(float(current['a0vx']), float(current['a0vy'])),
             arrowprops=dict(arrowstyle='->', color='darkblue', lw=2,
                             label='grad (noisy)'))
scale2 = lim_vel / max(abs(vg0x_fresh), abs(vg0y_fresh), 1e-9) * 0.3
ax2.annotate('', xy=(float(current['a0vx']) + vg0x_fresh * scale2,
                     float(current['a0vy']) + vg0y_fresh * scale2),
             xytext=(float(current['a0vx']), float(current['a0vy'])),
             arrowprops=dict(arrowstyle='->', color='lightblue', lw=2))
a0_action = int(float(current['action0']))
act_vec = action_tensor[a0_action].cpu().numpy()
ax2.annotate('', xy=(float(current['a0vx']) + act_vec[0] * lim_vel * 0.3,
                     float(current['a0vy']) + act_vec[1] * lim_vel * 0.3),
             xytext=(float(current['a0vx']), float(current['a0vy'])),
             arrowprops=dict(arrowstyle='->', color='white', lw=2))
ax2.legend()
ax2.set_xlabel('Agent0 vx'); ax2.set_ylabel('Agent0 vy')
a0x, a0y = float(current['a0x']), float(current['a0y'])
ax2.set_title(f'Value vs Velocity (pos=({a0x:.1f},{a0y:.1f}), frame {current["frame"]})')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('./simulation/value_map.pdf')
plt.close()
print('Saved to simulation/value_map.pdf')
