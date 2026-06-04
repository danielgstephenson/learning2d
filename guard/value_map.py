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

# State order: a0vel(0:2), a1vel(2:4), b0vel(4:6), b1vel(6:8),
#              a0pos(8:10), a1pos(10:12), b0pos(12:14), b1pos(14:16),
#              alive0(16), alive1(17), wp0(18:26), wp1(26:34)
ref_state = torch.zeros(1, 34, device=device)
ref_state[0, 0]  = float(current['a0vx'])
ref_state[0, 1]  = float(current['a0vy'])
ref_state[0, 2]  = float(current['a1vx'])
ref_state[0, 3]  = float(current['a1vy'])
ref_state[0, 4]  = float(current['b0vx'])
ref_state[0, 5]  = float(current['b0vy'])
ref_state[0, 6]  = float(current['b1vx'])
ref_state[0, 7]  = float(current['b1vy'])
ref_state[0, 8]  = float(current['a0x'])
ref_state[0, 9]  = float(current['a0y'])
ref_state[0, 10] = float(current['a1x'])
ref_state[0, 11] = float(current['a1y'])
ref_state[0, 12] = float(current['b0x'])
ref_state[0, 13] = float(current['b0y'])
ref_state[0, 14] = float(current['b1x'])
ref_state[0, 15] = float(current['b1y'])
ref_state[0, 16] = float(current['life0'])
ref_state[0, 17] = float(current['life1'])

wp_keys = ['a0wp0x','a0wp0y','a0wp1x','a0wp1y','a0wp2x','a0wp2y','a0wp3x','a0wp3y',
           'a1wp0x','a1wp0y','a1wp1x','a1wp1y','a1wp2x','a1wp2y','a1wp3x','a1wp3y']
for i, key in enumerate(wp_keys):
    ref_state[0, 18 + i] = float(current[key])

# Compute fresh gradients for both agents
gradient_fn = vmap(fgrad(lambda x: value_model(x).sum()))
fresh_grad = gradient_fn(ref_state)
vg0x_fresh = fresh_grad[0, 0].item()
vg0y_fresh = fresh_grad[0, 1].item()
vg1x_fresh = fresh_grad[0, 2].item()
vg1y_fresh = fresh_grad[0, 3].item()

N = 200
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
theta = np.linspace(0, 2*np.pi, 200)
ring_r = 13
agent_r = 5
blade_r = 10
lim_pos = 40
lim_vel = 1

# Reference positions for both agents (used across multiple plots)
a0x_ref, a0y_ref = float(current['a0x']), float(current['a0y'])
a1x_ref, a1y_ref = float(current['a1x']), float(current['a1y'])

# Arena walls
cx = [float(current[k]) for k in ['c0x','c1x','c2x','c3x','c0x']]
cy = [float(current[k]) for k in ['c0y','c1y','c2y','c3y','c0y']]

# --- Plot 0,0: Value vs Agent0 position ---
xs = np.linspace(a0x_ref - lim_pos, a0x_ref + lim_pos, N)
ys = np.linspace(a0y_ref - lim_pos, a0y_ref + lim_pos, N)
XX, YY = np.meshgrid(xs, ys)
states = ref_state.repeat(N * N, 1)
states[:, 8:10] = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=1),
                                dtype=torch.float32, device=device)
with torch.no_grad():
    val_pos0 = torch.sigmoid(value_model(states)).cpu().numpy().reshape(N, N)

vmin, vmax = val_pos0.min(), val_pos0.max()
print(f'Agent0 position map: [{vmin:.4f}, {vmax:.4f}]')
np.savetxt('./simulation/value_pos.csv',
           np.column_stack([XX.ravel(), YY.ravel(), val_pos0.ravel()]),
           delimiter=',', header='x,y,value', comments='')
ax = axes[0, 0]
im = ax.imshow(val_pos0, extent=[a0x_ref-lim_pos, a0x_ref+lim_pos, a0y_ref-lim_pos, a0y_ref+lim_pos],
               origin='lower', cmap='RdYlGn', vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax, label='Value')
ax.plot(ring_r * np.cos(theta), ring_r * np.sin(theta), 'k-', lw=2)
# Own blade (blade0) — dotted
b0x_ref, b0y_ref = float(current['b0x']), float(current['b0y'])
ax.plot(b0x_ref + blade_r * np.cos(theta), b0y_ref + blade_r * np.sin(theta),
        color='blue', lw=1.5, linestyle=':', label='Blade0')
b0vx_val, b0vy_val = float(current['b0vx']), float(current['b0vy'])
b0_scale = lim_pos * 0.3 / max(abs(b0vx_val), abs(b0vy_val), 1e-9)
ax.annotate('', xy=(b0x_ref + b0vx_val * b0_scale, b0y_ref + b0vy_val * b0_scale),
            xytext=(b0x_ref, b0y_ref), annotation_clip=False,
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, linestyle='dotted'))
# Enemy blade (blade1) — solid
b1x_ref, b1y_ref = float(current['b1x']), float(current['b1y'])
ax.plot(b1x_ref + blade_r * np.cos(theta), b1y_ref + blade_r * np.sin(theta), 'b-', lw=1.5, label='Blade1')
b1vx_val, b1vy_val = float(current['b1vx']), float(current['b1vy'])
b1_scale = lim_pos * 0.3 / max(abs(b1vx_val), abs(b1vy_val), 1e-9)
ax.annotate('', xy=(b1x_ref + b1vx_val * b1_scale, b1y_ref + b1vy_val * b1_scale),
            xytext=(b1x_ref, b1y_ref), annotation_clip=False,
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
# Self
ax.plot([a0x_ref, b0x_ref], [a0y_ref, b0y_ref], color='blue', lw=1.0, linestyle=':')
ax.plot(a0x_ref + agent_r * np.cos(theta), a0y_ref + agent_r * np.sin(theta),
        color='blue', lw=1.5, linestyle='--')
ax.plot(a0x_ref, a0y_ref, 'bo', ms=8, markeredgecolor='k', label='Agent0')
ax.annotate('', xy=(a0x_ref + float(current['a0vx']), a0y_ref + float(current['a0vy'])),
            xytext=(a0x_ref, a0y_ref),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
# Other agent (agent1)
ax.plot([a1x_ref, b1x_ref], [a1y_ref, b1y_ref], color='blue', lw=1.0, linestyle='-')
ax.plot(a1x_ref + agent_r * np.cos(theta), a1y_ref + agent_r * np.sin(theta),
        color='blue', lw=1.0, linestyle='-')
ax.plot(a1x_ref, a1y_ref, 'b^', ms=8, markeredgecolor='k', label='Agent1')
ax.annotate('', xy=(a1x_ref + float(current['a1vx']), a1y_ref + float(current['a1vy'])),
            xytext=(a1x_ref, a1y_ref),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.plot(cx, cy, 'k-', lw=1.5, zorder=0)
ax.set_xlabel('Agent0 x'); ax.set_ylabel('Agent0 y')
ax.set_title(f'Value vs Agent0 Position (frame {current["frame"]})')
ax.set_xlim(a0x_ref-lim_pos, a0x_ref+lim_pos)
ax.set_ylim(a0y_ref-lim_pos, a0y_ref+lim_pos)
ax.legend(); ax.set_aspect('equal')

# --- Plot 0,1: Value vs Agent0 velocity ---
a0vx_ref, a0vy_ref = float(current['a0vx']), float(current['a0vy'])
vs_x = np.linspace(a0vx_ref - lim_vel, a0vx_ref + lim_vel, N)
vs_y = np.linspace(a0vy_ref - lim_vel, a0vy_ref + lim_vel, N)
VX, VY = np.meshgrid(vs_x, vs_y)
states2 = ref_state.repeat(N * N, 1)
states2[:, 0:2] = torch.tensor(np.stack([VX.ravel(), VY.ravel()], axis=1),
                                dtype=torch.float32, device=device)
with torch.no_grad():
    val_vel0 = torch.sigmoid(value_model(states2)).cpu().numpy().reshape(N, N)

vmin2, vmax2 = val_vel0.min(), val_vel0.max()
print(f'Agent0 velocity map: [{vmin2:.4f}, {vmax2:.4f}]')
np.savetxt('./simulation/value_vel.csv',
           np.column_stack([VX.ravel(), VY.ravel(), val_vel0.ravel()]),
           delimiter=',', header='vx,vy,value', comments='')
ax2 = axes[0, 1]
im2 = ax2.imshow(val_vel0, extent=[a0vx_ref-lim_vel, a0vx_ref+lim_vel, a0vy_ref-lim_vel, a0vy_ref+lim_vel],
                 origin='lower', cmap='RdYlGn', vmin=vmin2, vmax=vmax2)
plt.colorbar(im2, ax=ax2, label='Value')
ax2.axhline(a0vy_ref, color='k', lw=0.5); ax2.axvline(a0vx_ref, color='k', lw=0.5)
ax2.plot(a0vx_ref, a0vy_ref, 'bo', ms=8, markeredgecolor='k', label='Agent0 vel')
vg0x, vg0y = float(current['vg0x']), float(current['vg0y'])
scale = lim_vel / max(abs(vg0x), abs(vg0y), 1e-9) * 0.3
ax2.annotate('', xy=(a0vx_ref + vg0x * scale, a0vy_ref + vg0y * scale),
             xytext=(a0vx_ref, a0vy_ref),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2, linestyle='dotted'))
scale2 = lim_vel / max(abs(vg0x_fresh), abs(vg0y_fresh), 1e-9) * 0.3
ax2.annotate('', xy=(a0vx_ref + vg0x_fresh * scale2, a0vy_ref + vg0y_fresh * scale2),
             xytext=(a0vx_ref, a0vy_ref),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2, linestyle='dashed'))
a0_action = int(float(current['action0']))
act_vec0 = action_tensor[a0_action].cpu().numpy()
ax2.annotate('', xy=(a0vx_ref + act_vec0[0] * lim_vel * 0.3, a0vy_ref + act_vec0[1] * lim_vel * 0.3),
             xytext=(a0vx_ref, a0vy_ref),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2, linestyle='solid'))
ax2.plot([], [], color='blue', lw=2, linestyle='solid', label='Action')
ax2.plot([], [], color='blue', lw=2, linestyle='dashed', label='Fresh gradient')
ax2.plot([], [], color='blue', lw=2, linestyle='dotted', label='Noisy gradient')
ax2.legend()
ax2.set_xlabel('Agent0 vx'); ax2.set_ylabel('Agent0 vy')
ax2.set_title(f'Value vs Agent0 Velocity (pos=({a0x_ref:.1f},{a0y_ref:.1f}), frame {current["frame"]})')
ax2.set_aspect('equal')

# --- Plot 1,0: Value vs Agent1 position ---
xs1 = np.linspace(a1x_ref - lim_pos, a1x_ref + lim_pos, N)
ys1 = np.linspace(a1y_ref - lim_pos, a1y_ref + lim_pos, N)
XX1, YY1 = np.meshgrid(xs1, ys1)
states3 = ref_state.repeat(N * N, 1)
states3[:, 10:12] = torch.tensor(np.stack([XX1.ravel(), YY1.ravel()], axis=1),
                                  dtype=torch.float32, device=device)
with torch.no_grad():
    val_pos1 = torch.sigmoid(value_model(states3)).cpu().numpy().reshape(N, N)

vmin3, vmax3 = val_pos1.min(), val_pos1.max()
print(f'Agent1 position map: [{vmin3:.4f}, {vmax3:.4f}]')
np.savetxt('./simulation/value_pos1.csv',
           np.column_stack([XX1.ravel(), YY1.ravel(), val_pos1.ravel()]),
           delimiter=',', header='x,y,value', comments='')
ax3 = axes[1, 0]
im3 = ax3.imshow(val_pos1, extent=[a1x_ref-lim_pos, a1x_ref+lim_pos, a1y_ref-lim_pos, a1y_ref+lim_pos],
                 origin='lower', cmap='RdYlGn_r', vmin=vmin3, vmax=vmax3)
plt.colorbar(im3, ax=ax3, label='Value')
ax3.plot(ring_r * np.cos(theta), ring_r * np.sin(theta), 'k-', lw=2)
# Own blade (blade1) — dotted
b1x_ref2, b1y_ref2 = float(current['b1x']), float(current['b1y'])
ax3.plot(b1x_ref2 + blade_r * np.cos(theta), b1y_ref2 + blade_r * np.sin(theta),
         color='blue', lw=1.5, linestyle=':', label='Blade1')
b1vx_val2, b1vy_val2 = float(current['b1vx']), float(current['b1vy'])
b1_scale2 = lim_pos * 0.3 / max(abs(b1vx_val2), abs(b1vy_val2), 1e-9)
ax3.annotate('', xy=(b1x_ref2 + b1vx_val2 * b1_scale2, b1y_ref2 + b1vy_val2 * b1_scale2),
             xytext=(b1x_ref2, b1y_ref2), annotation_clip=False,
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, linestyle='dotted'))
# Enemy blade (blade0) — solid
b0x_ref, b0y_ref = float(current['b0x']), float(current['b0y'])
ax3.plot(b0x_ref + blade_r * np.cos(theta), b0y_ref + blade_r * np.sin(theta), 'b-', lw=1.5, label='Blade0')
b0vx_val, b0vy_val = float(current['b0vx']), float(current['b0vy'])
b0_scale = lim_pos * 0.3 / max(abs(b0vx_val), abs(b0vy_val), 1e-9)
ax3.annotate('', xy=(b0x_ref + b0vx_val * b0_scale, b0y_ref + b0vy_val * b0_scale),
             xytext=(b0x_ref, b0y_ref), annotation_clip=False,
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
# Self
ax3.plot([a1x_ref, b1x_ref2], [a1y_ref, b1y_ref2], color='blue', lw=1.0, linestyle=':')
ax3.plot(a1x_ref + agent_r * np.cos(theta), a1y_ref + agent_r * np.sin(theta),
         color='blue', lw=1.5, linestyle='--')
ax3.plot(a1x_ref, a1y_ref, 'bo', ms=8, markeredgecolor='k', label='Agent1')
ax3.annotate('', xy=(a1x_ref + float(current['a1vx']), a1y_ref + float(current['a1vy'])),
             xytext=(a1x_ref, a1y_ref),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
# Other agent (agent0)
ax3.plot([a0x_ref, b0x_ref], [a0y_ref, b0y_ref], color='blue', lw=1.0, linestyle='-')
ax3.plot(a0x_ref + agent_r * np.cos(theta), a0y_ref + agent_r * np.sin(theta),
         color='blue', lw=1.0, linestyle='-')
ax3.plot(a0x_ref, a0y_ref, 'b^', ms=8, markeredgecolor='k', label='Agent0')
ax3.annotate('', xy=(a0x_ref + float(current['a0vx']), a0y_ref + float(current['a0vy'])),
             xytext=(a0x_ref, a0y_ref),
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax3.plot(cx, cy, 'k-', lw=1.5, zorder=0)
ax3.set_xlabel('Agent1 x'); ax3.set_ylabel('Agent1 y')
ax3.set_title(f'Value vs Agent1 Position (frame {current["frame"]})')
ax3.set_xlim(a1x_ref-lim_pos, a1x_ref+lim_pos)
ax3.set_ylim(a1y_ref-lim_pos, a1y_ref+lim_pos)
ax3.legend(); ax3.set_aspect('equal')

# --- Plot 1,1: Value vs Agent1 velocity ---
a1vx_ref, a1vy_ref = float(current['a1vx']), float(current['a1vy'])
vs1_x = np.linspace(a1vx_ref - lim_vel, a1vx_ref + lim_vel, N)
vs1_y = np.linspace(a1vy_ref - lim_vel, a1vy_ref + lim_vel, N)
VX1, VY1 = np.meshgrid(vs1_x, vs1_y)
states4 = ref_state.repeat(N * N, 1)
states4[:, 2:4] = torch.tensor(np.stack([VX1.ravel(), VY1.ravel()], axis=1),
                                dtype=torch.float32, device=device)
with torch.no_grad():
    val_vel1 = torch.sigmoid(value_model(states4)).cpu().numpy().reshape(N, N)

vmin4, vmax4 = val_vel1.min(), val_vel1.max()
print(f'Agent1 velocity map: [{vmin4:.4f}, {vmax4:.4f}]')
np.savetxt('./simulation/value_vel1.csv',
           np.column_stack([VX1.ravel(), VY1.ravel(), val_vel1.ravel()]),
           delimiter=',', header='vx,vy,value', comments='')
ax4 = axes[1, 1]
im4 = ax4.imshow(val_vel1, extent=[a1vx_ref-lim_vel, a1vx_ref+lim_vel, a1vy_ref-lim_vel, a1vy_ref+lim_vel],
                 origin='lower', cmap='RdYlGn_r', vmin=vmin4, vmax=vmax4)
plt.colorbar(im4, ax=ax4, label='Value')
ax4.axhline(a1vy_ref, color='k', lw=0.5); ax4.axvline(a1vx_ref, color='k', lw=0.5)
ax4.plot(a1vx_ref, a1vy_ref, 'bo', ms=8, markeredgecolor='k', label='Agent1 vel')
vg1x, vg1y = float(current['vg1x']), float(current['vg1y'])
scale3 = lim_vel / max(abs(vg1x), abs(vg1y), 1e-9) * 0.3
ax4.annotate('', xy=(a1vx_ref - vg1x * scale3, a1vy_ref - vg1y * scale3),
             xytext=(a1vx_ref, a1vy_ref),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2, linestyle='dotted'))
scale4 = lim_vel / max(abs(vg1x_fresh), abs(vg1y_fresh), 1e-9) * 0.3
ax4.annotate('', xy=(a1vx_ref - vg1x_fresh * scale4, a1vy_ref - vg1y_fresh * scale4),
             xytext=(a1vx_ref, a1vy_ref),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2, linestyle='dashed'))
a1_action = int(float(current['action1']))
act_vec1 = action_tensor[a1_action].cpu().numpy()
ax4.annotate('', xy=(a1vx_ref + act_vec1[0] * lim_vel * 0.3, a1vy_ref + act_vec1[1] * lim_vel * 0.3),
             xytext=(a1vx_ref, a1vy_ref),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2, linestyle='solid'))
ax4.plot([], [], color='blue', lw=2, linestyle='solid', label='Action')
ax4.plot([], [], color='blue', lw=2, linestyle='dashed', label='Fresh gradient')
ax4.plot([], [], color='blue', lw=2, linestyle='dotted', label='Noisy gradient')
ax4.legend()
ax4.set_xlabel('Agent1 vx'); ax4.set_ylabel('Agent1 vy')
ax4.set_title(f'Value vs Agent1 Velocity (pos=({a1x_ref:.1f},{a1y_ref:.1f}), frame {current["frame"]})')
ax4.set_aspect('equal')

plt.tight_layout()
plt.savefig('./simulation/value_map.pdf')
plt.close()
print('Saved to simulation/value_map.pdf')
