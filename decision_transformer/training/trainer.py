import numpy as np
import torch

import time
from torch.utils.tensorboard import SummaryWriter
# loggerdir = "./tb_record_normal_hopper_ori"
# loggerdir = "./tb_record_expert_hopper_ori"
# loggerdir = "./tb_record_replay_hopper_ori"
# loggerdir = "./tb_record_normal_hopper_dwa"
# loggerdir = "./tb_record_expert_hopper_dwa"
# loggerdir = "./tb_record_replay_hopper_dwa"
# loggerdir = "./tb_record_normal_halfcheetah_ori"
# loggerdir = "./tb_record_expert_halfcheetah_ori"
# loggerdir = "./tb_record_replay_halfcheetah_ori"
# loggerdir = "./tb_record_normal_halfcheetah_swa"
# loggerdir = "./tb_record_expert_halfcheetah_swa"
# loggerdir = "./tb_record_replay_halfcheetah_swa"
# loggerdir = "./tb_record_normal_hopper_swa_with_noise"
# loggerdir = "./tb_record_expert_hopper_swa_with_noise"
loggerdir = "./tb_record_replay_hopper_swa_with_noise"

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        writer = SummaryWriter(loggerdir)

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
        #import pudb; pudb.set_trace()
        self.optimizer.swap_swa_sgd()
        # Reset swa.
        for group in self.optimizer.param_groups:
            group['n_avg'] = 0
            group['step_counter'] = 0
        # Since SWA resets batch norm statistics, recalculate it doing one inference.
        train_loss = self.train_step()
        train_losses.append(train_loss)


        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        writer.add_scalar('time/training', time.time() - train_start, iter_num)
        writer.add_scalar('time/total', time.time() - self.start_time, iter_num)
        writer.add_scalar('time/evaluation', time.time() - eval_start, iter_num)
        writer.add_scalar('training/train_loss_mean', np.mean(train_losses), iter_num)
        writer.add_scalar('training/train_loss_std', np.std(train_losses), iter_num)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
                writer.add_scalar(k, v, iter_num)

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()