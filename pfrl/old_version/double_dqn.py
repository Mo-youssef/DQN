import torch
import dqn
from pfrl.utils import evaluating
from pfrl.utils.recurrent import pack_and_forward
# import pdb

class DoubleDQN(dqn.DQN):
    """Double DQN.

    See: http://arxiv.org/abs/1509.06461.
    """

    def _compute_target_values(self, exp_batch):
        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]
        if self.rnd_reward:
            # pdb.set_trace()
            batch_current_states = exp_batch["state"]
            rnd_targets = self.rnd_models['target'](batch_current_states)
            with torch.enable_grad():
                rnd_preds = self.rnd_models['predict'](batch_current_states)
                rnd_rewards = ((rnd_targets - rnd_preds)**2).sum(axis=1)**0.5
                self.rnd_models['optimizer'].zero_grad()
                mean_reward = rnd_rewards.sum()
                mean_reward.backward()
                self.mean_intrinsic_reward = mean_reward.item() / batch_current_states.shape[0]
                self.rnd_models['optimizer'].step()
            rnd_rewards_detached = rnd_rewards.detach()
            self.moving_std.push(rnd_rewards_detached.cpu().numpy())
            # batch_rewards += rnd_rewards_detached / torch.as_tensor(self.moving_std.std(), device=self.device, dtype=torch.float32) 
            batch_rewards += rnd_rewards_detached / self.moving_std.std()
            
        batch_next_state = exp_batch["next_state"]

        with evaluating(self.model):
            if self.recurrent:
                next_qout, _ = pack_and_forward(
                    self.model, batch_next_state, exp_batch["next_recurrent_state"],
                )
            else:
                next_qout = self.model(batch_next_state)

        if self.recurrent:
            target_next_qout, _ = pack_and_forward(
                self.target_model, batch_next_state, exp_batch["next_recurrent_state"],
            )
        else:
            target_next_qout = self.target_model(batch_next_state)

        next_q_max = target_next_qout.evaluate_actions(next_qout.greedy_actions)

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max
