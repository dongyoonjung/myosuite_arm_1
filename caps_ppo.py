"""CAPS-PPO — Conditioning for Action Policy Smoothness (Mysore+ 2021) 위 PPO.

anti-tremor 스택 최강 계층(DESIGN escalation #2, 팔꿈치 caps_ppo 계승).
PPO 손실에 공간 평활항 추가:
    L_S = λ · E ||μ(s) − μ(s+ε)||² ,  ε~N(0, σ)
근접 상태가 비슷한 행동을 내도록 정책을 정칙화 → 고주파 떨림 억제.
(시간 평활은 env의 leaky-integral·dctrl 보상이 담당.)

train.py에서 --caps 플래그로 사용(별도 배선). 단독으로도:
    from caps_ppo import CAPSPPO
    model = CAPSPPO("MlpPolicy", venv, caps_lambda=0.5, caps_sigma=0.05, ...)
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class CAPSPPO(PPO):
    def __init__(self, *args, caps_lambda=0.5, caps_sigma=0.05, **kw):
        self.caps_lambda = caps_lambda
        self.caps_sigma = caps_sigma
        super().__init__(*args, **kw)

    def _mean_action(self, obs):
        """관측에서 정책 평균행동(tanh 전 가우시안 평균)."""
        dist = self.policy.get_distribution(obs)
        return dist.distribution.mean

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, caps_losses = [], [], [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # --- CAPS 공간 평활 ---
                obs = rollout_data.observations
                mu = self._mean_action(obs)
                noise = th.randn_like(obs) * self.caps_sigma
                mu_near = self._mean_action(obs + noise)
                caps_loss = ((mu - mu_near) ** 2).sum(dim=1).mean()
                caps_losses.append(caps_loss.item())

                loss = (policy_loss + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss + self.caps_lambda * caps_loss)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                            self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/caps_loss", np.mean(caps_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
