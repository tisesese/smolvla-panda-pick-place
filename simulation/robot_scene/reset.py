"""Original ResetPlacement class, separated from the historical collection pipeline."""
import numpy as np
import mujoco
from .scripted_grasp import contact_metrics

class ResetPlacement:
    def __init__(self, seed, xyz, draw):
        self.seed, self.xyz, self.draw = seed, np.asarray(xyz), draw
        self.metadata = {}
        self.eq_active = None
        self.settled = False
        self.unexpected_collision_ticks = 0
        self.unexpected_collision_pairs = set()

    def __call__(self, scene):
        model, data = scene.model, scene.data
        assert data.time == 0.0
        jid = model.joint('cube_free').id
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
        qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        table = data.geom('table_top')
        size = model.geom('table_top').size
        cube_size = model.geom('red_cube_geom').size
        top = float(table.xpos[2] + size[2])
        assert np.all(np.abs(self.xyz[:2] - table.xpos[:2]) + cube_size[:2] < size[:2] - .05)
        assert self.xyz[2] - cube_size[2] >= top + .001
        # The only live cube qpos write: legal reset, t=0, prior to all physics.
        data.qpos[qadr:qadr + 7] = np.r_[self.xyz, [1., 0., 0., 0.]]
        data.qvel[dadr:dadr + 6] = 0
        mujoco.mj_forward(model, data)
        cube_bid = model.body('red_cube').id
        cube_contacts = []
        for c in data.contact:
            if cube_bid in (int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])):
                cube_contacts.append({'geoms': [model.geom(c.geom1).name, model.geom(c.geom2).name],
                                      'distance_m': float(c.dist)})
        assert not cube_contacts, f'Unexpected initial cube collision: {cube_contacts}'
        self.eq_active = data.eq_active.copy()
        self.metadata.update({'seed': self.seed, 'rng_draw_index': self.draw,
            'requested_xyz_m': self.xyz.tolist(), 'actual_reset_xyz_m': data.body('red_cube').xpos.tolist(),
            'free_joint_id': int(jid), 'qpos_address': qadr, 'dof_address': dadr,
            'table_top_z_m': top, 'table_xy_half_extents_m': size[:2].tolist(),
            'cube_half_extents_m': cube_size.tolist(), 'initial_bottom_clearance_m': float(self.xyz[2] - cube_size[2] - top),
            'initial_cube_contacts': cube_contacts, 'simulation_time_at_write': float(data.time),
            'writes_during_execution': 0, 'existing_equality_count': int(model.neq)})
        return self.metadata

    def before_step(self, scene, interface, phase, action, ctrl, physics_steps):
        assert np.array_equal(scene.data.eq_active, self.eq_active)
        assert not np.any(scene.data.qfrc_applied) and not np.any(scene.data.xfrc_applied)
        if not self.settled and phase == 'approach':
            cube = scene.data.body('red_cube').xpos.copy()
            jid = scene.model.joint('cube_free').id
            dadr = int(scene.model.jnt_dofadr[jid])
            velocity = scene.data.qvel[dadr:dadr + 6]
            expected_z = self.metadata['table_top_z_m'] + scene.model.geom('red_cube_geom').size[2]
            stable = (np.linalg.norm(cube[:2] - self.xyz[:2]) < .001 and
                      abs(cube[2] - expected_z) < .002 and np.linalg.norm(velocity) < .01)
            self.metadata['settled'] = {'xyz_m': cube.tolist(), 'qvel': velocity.tolist(),
                'time_s': float(scene.data.time), 'stable': bool(stable), 'contact': contact_metrics(scene)}
            self.settled = True
            if not stable:
                raise RuntimeError('reset_placement_unstable')
        collision = False
        for c in scene.data.contact:
            names = [scene.model.geom(c.geom1).name, scene.model.geom(c.geom2).name]
            bodies = [scene.model.body(int(scene.model.geom_bodyid[g])).name for g in (c.geom1, c.geom2)]
            if 'table_top' in names or 'floor' in names:
                if any(b.startswith('link') and b != 'link0' or b in ('hand', 'left_finger', 'right_finger') for b in bodies):
                    collision = True
                    self.unexpected_collision_pairs.add(tuple(sorted(str(n) for n in names)))
        self.unexpected_collision_ticks += int(collision)
