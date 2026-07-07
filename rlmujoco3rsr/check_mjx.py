"""
Quick MJX sanity check for the 3-RSR closed-loop model.
Run on Ubuntu + GPU after installing: mujoco, jax[cuda], mujoco-mjx.

    python check_mjx.py
"""
import time
import numpy as np
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

XML = "three_leg_3rsr_closed.xml"

def main():
    print("JAX backend:", jax.default_backend())
    print("devices:", jax.devices())

    # load on CPU then move to device
    m = mujoco.MjModel.from_xml_path(XML)
    print(f"CPU model OK: nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq}")

    mx = mjx.put_model(m)
    dx = mjx.make_data(mx)

    # set a control command: hip motors 0..2, servos/thrust held at 0 (nu=9 total)
    ctrl = jp.zeros(m.nu).at[0:3].set(jp.array([-1.2, -0.3, 0.4]))
    dx = dx.replace(ctrl=ctrl)

    step = jax.jit(mjx.step)

    # warmup + time a few steps
    t0 = time.time()
    for _ in range(200):
        dx = step(mx, dx)
    dx.qpos.block_until_ready()
    print(f"200 MJX steps OK in {time.time()-t0:.2f}s (incl. compile)")

    # batched rollout (this is what RL uses: many envs in parallel)
    N = 2048
    batch = jax.vmap(lambda _: mjx.make_data(mx).replace(ctrl=ctrl))(jp.arange(N))
    bstep = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    t0 = time.time()
    for _ in range(50):
        batch = bstep(mx, batch)
    batch.qpos.block_until_ready()
    dt = time.time()-t0
    print(f"{N} envs x 50 steps in {dt:.2f}s  -> {N*50/dt:,.0f} steps/s")

    print("MJX closed-loop (equality connect) runs fine. Ready for RL.")

if __name__ == "__main__":
    main()
