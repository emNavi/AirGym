    def compute_ref_pos(self):
        '''
        Four control points define the cubic spline function:
        Segment 1:
        x(t) = 6.0*t
        y(t) = -36.0*t**3 + 10.0*t - 1.0
        z(t) = 1

        Segment 2:
        x(t) = 6.0*t + 2.0
        y(t) = 72.0*t**3 - 36.0*t**2 - 2.0*t + 1.0
        z(t) = 1

        Segment 3:
        x(t) = 6.0*t + 4.0
        y(t) = -36.0*t**3 + 36.0*t**2 - 2.0*t - 1.0
        z(t) = 1
        '''
        horizon = self.max_episode_length
        
        t = torch.zeros_like(self.progress_buf, dtype=torch.float32)
        cond1 = (self.progress_buf >= 0) & (self.progress_buf < horizon/4)
        cond2 = (self.progress_buf >= horizon/4) & (self.progress_buf < horizon/2)
        cond3 = (self.progress_buf >= horizon/2) & (self.progress_buf < 3*horizon/4)
        cond4 = (self.progress_buf >= 3*horizon/4) & (self.progress_buf <= horizon)

        t[cond1] = (self.progress_buf[cond1] - 0) / (horizon/4) / 3
        t[cond2] = (self.progress_buf[cond2] - horizon/4) / (horizon/4) / 3
        t[cond3] = (self.progress_buf[cond3] - horizon/2) / (horizon/4) / 3

        x1 = 6*t
        y1 = -36*t**3 + 10*t - 1
        z1 = torch.full_like(t, 1)

        x2 = 6*t + 2
        y2 = 72*t**3 - 36*t**2 - 2*t + 1
        z2 = torch.full_like(t, 1)

        x3 = 6*t + 4
        y3 = -36*t**3 + 36*t**2 - 2*t - 1
        z3 = torch.full_like(t, 1)

        x4 = torch.full_like(t, 6)
        y4 = torch.full_like(t, 1)
        z4 = torch.full_like(t, 1)

        x = torch.where(cond1, x1, torch.where(cond2, x2, torch.where(cond3, x3, x4)))
        y = torch.where(cond1, y1, torch.where(cond2, y2, torch.where(cond3, y3, y4)))
        z = torch.where(cond1, z1, torch.where(cond2, z2, torch.where(cond3, z3, z4)))

        pose = torch.stack((x, y, z), dim=-1)
        return pose
    
    def compute_ref_vel(self, root_positions, root_linvels):
        '''
        Four control points define the velocity direction
        '''
        horizon = self.max_episode_length

        t = torch.zeros_like(self.progress_buf, dtype=torch.float32)
        cond1 = (self.progress_buf >= 0) & (self.progress_buf < horizon/4)
        cond2 = (self.progress_buf >= horizon/4) & (self.progress_buf < horizon/2)
        cond3 = (self.progress_buf >= horizon/2) & (self.progress_buf < 3*horizon/4)
        cond4 = (self.progress_buf >= 3*horizon/4) & (self.progress_buf <= horizon)

        x1 = 2
        y1 = 1
        z1 = 1

        x2 = 4
        y2 = -1
        z2 = 1

        x3 = 6
        y3 = 1
        z3 = 1

        x4 = 6
        y4 = 1
        z4 = 1
        
        x = torch.where(cond1, x1, torch.where(cond2, x2, torch.where(cond3, x3, x4)))
        y = torch.where(cond1, y1, torch.where(cond2, y2, torch.where(cond3, y3, y4)))
        z = torch.where(cond1, z1, torch.where(cond2, z2, torch.where(cond3, z3, z4)))

        ref_pos = torch.stack((x, y, z), dim=-1)
        ref_direction = ref_pos - root_positions  # (N, 3)

        ref_direction = ref_direction / torch.norm(ref_direction, dim=-1, keepdim=True).clamp(min=1e-8)
        root_linvels = root_linvels / torch.norm(root_linvels, dim=-1, keepdim=True).clamp(min=1e-8)

        similarity = torch.sum(ref_direction * root_linvels, dim=-1)  # (N,) (-1~1)
        return similarity
    
    def guidance_reward(self, root_positions, pre_root_positions, root_angvels):
        target = torch.tensor([6, 1, 1], device=self.device).repeat(self.num_envs, 1)
        r = torch.norm(target-pre_root_positions, dim=-1) - torch.norm(target-root_positions, dim=-1) - 0.01 * torch.norm(root_angvels, dim=-1)
        return r

    def compute_quadcopter_reward(self, root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length):
        # ref_pos = self.compute_ref_pos()
        # print(progress_buf, ":", ref_pos[0])
        # diff = root_positions - ref_pos
        # dist = torch.sqrt(diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1] + (diff[..., 2]) * (diff[..., 2]))
        # pos_reward = 2 * (1.0 - (1/4)*dist)

        guidance_reward = self.guidance_reward(root_positions, self.pre_root_positions, root_angvels)
        # print(root_positions[0], self.pre_root_positions[0])

        # print("guidance_reward:", guidance_reward[0])

        # reward = guidance_reward

        # resets due to misbehavior
        ones = torch.ones_like(reset_buf)
        die = torch.zeros_like(reset_buf)
        # die = torch.where(target_dist > 10.0, ones, die)

        # resets due to episode length
        reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
        # reset = torch.where(dist > 1.0, ones, reset)

        reset = torch.where(root_positions[..., 2] < 0.5, ones, reset)
        reset = torch.where(root_positions[..., 2] > 1.5, ones, reset)
        
        item_reward_info = {}
        # item_reward_info["pos_reward"] = pos_reward

        return reward, reset, item_reward_info