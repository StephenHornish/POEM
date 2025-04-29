from stable_baselines3 import PPO

import copy
import torch
import torch.nn.functional as F
from stable_baselines3.common.utils import explained_variance
from gym import spaces  

class POEM(PPO):
    def __init__(self, *args, kl_threshold=0.1, sigma_min=0.01, sigma_max=0.1, 
                 beta=0.9, lambda_diversity=0.1, **kwargs):
        self.kl_threshold = kl_threshold      
        self.sigma_min = sigma_min            
        self.sigma_max = sigma_max            
        self.beta = beta                      
        self.lambda_diversity = lambda_diversity  
        self.policy_kwargs = kwargs.get("policy_kwargs", {})
        self.theta_avg = None  # Will be set in _setup_model
        super().__init__(*args, **kwargs)
    def _setup_model(self):
        super()._setup_model()
        # Now that self.policy exists, initialize theta_avg
        self.theta_avg = copy.deepcopy(self.policy.state_dict())

    def compute_loss(self, policy):
        # Retrieve one batch of data from the rollout buffer
        data = next(iter(self.rollout_buffer.get(self.batch_size)))
        
        # Handle discrete actions if necessary
        if isinstance(self.action_space, spaces.Discrete):
            actions = data.actions.long().flatten()
        else:
            actions = data.actions

        # Evaluate actions using the given policy
        values, log_prob, entropy = policy.evaluate_actions(data.observations, actions)
        values = values.flatten()
        
        # Normalize advantages if needed
        advantages = data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute the ratio between new and old policy probabilities
        current_clip_range = self.clip_range(self._current_progress_remaining)
        ratio = torch.exp(log_prob - data.old_log_prob)
        
        # Compute the policy loss components
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - current_clip_range, 1 + current_clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Compute value function loss (with optional clipping)
        if self.clip_range_vf is None:
            values_pred = values
        else:
            values_pred = data.old_values + torch.clamp(
                values - data.old_values, -current_clip_range, current_clip_range
            )
        value_loss = torch.nn.functional.mse_loss(data.returns, values_pred)
        
        # Entropy loss for exploration encouragement
        entropy_loss = -torch.mean(entropy if entropy is not None else -log_prob)
        
        # Diversity bonus based on KL divergence
        kl_divergence = self.compute_kl_divergence(policy, self.theta_avg)
        loss_diversity = -self.lambda_diversity * kl_divergence
        
        # Total loss
        total_loss = policy_loss + loss_diversity + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        return total_loss


    def train(self):
        """
        Modified PPO training loop with adaptive exploration.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Handle discrete actions conversion if needed
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                else:
                    actions = rollout_data.actions

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                # Normalize advantage if applicable
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Standard PPO loss computation
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                # Compute KL divergence between current policy and the moving-average policy
                kl_divergence = self.compute_kl_divergence(self.policy, self.theta_avg)

                # Include diversity bonus into the PPO loss
                loss_diversity = -self.lambda_diversity * kl_divergence
                loss = policy_loss + loss_diversity

                # Compute value loss (with optional clipping)
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range, clip_range
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss for exploration encouragement
                entropy_loss = -torch.mean(entropy if entropy is not None else -log_prob)
                entropy_losses.append(entropy_loss.item())

                total_loss = loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Zero gradients, perform backward pass and update parameters
                self.policy.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Now, outside the gradient computation, perform the mutation mechanism.
                with torch.no_grad():
                    # Recompute KL divergence after the update
                    kl_divergence_post = self.compute_kl_divergence(self.policy, self.theta_avg)
                    if kl_divergence_post < self.kl_threshold:
                        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (
                            self.kl_threshold - kl_divergence_post
                        ) / self.kl_threshold

                        sigma = max(
                            self.sigma_min,
                            min(self.sigma_max, sigma)
                        )

                        perturbed_params = {
                            k: v + torch.normal(0, sigma, size=v.shape).to(v.device)
                            for k, v in self.policy.state_dict().items()
                        }

                        mutated_policy = self._create_policy_instance()
                        mutated_policy.load_state_dict(perturbed_params)

                        mutated_loss = self.compute_loss(mutated_policy)
                        current_loss = self.compute_loss(self.policy)

                        # If the mutated policy has a better loss, adopt its parameters
                        if mutated_loss < current_loss:
                            self.policy.load_state_dict(perturbed_params)

                    # Log approximate KL divergence
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # Update the moving average of policy parameters (after mutation update)
                with torch.no_grad():
                    current_params = self.policy.state_dict()
                    for k in self.theta_avg:
                        self.theta_avg[k] = self.beta * self.theta_avg[k] + (1 - self.beta) * current_params[k]

            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Log metrics
        self.logger.record("train/entropy_loss", torch.tensor(entropy_losses).mean().item())
        self.logger.record("train/policy_gradient_loss", torch.tensor(pg_losses).mean().item())
        self.logger.record("train/value_loss", torch.tensor(value_losses).mean().item())
        self.logger.record("train/approx_kl", torch.tensor(approx_kl_divs).mean().item())
        self.logger.record("train/loss", total_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/clip_range", clip_range)

    def compute_kl_divergence(self, policy1, policy_avg_params):
        # Build the average-policy model
        policy_avg_model = self._create_policy_instance()
        policy_avg_model.load_state_dict(policy_avg_params)

        # log_probs1, log_probs2 on a batch of observations+actions:
        log_probs1 = self.get_log_probs(policy1)
        log_probs2 = self.get_log_probs(policy_avg_model)

        # Exponentiate log_probs1 to get p
        p1 = torch.exp(log_probs1)
        # Now compute the usual KL(p||q) = sum( p1 * (log_probs1 - log_probs2) )
        kl_div = (p1 * (log_probs1 - log_probs2)).sum(dim=-1).mean()
        return kl_div


    def _create_policy_instance(self):
        """
        Create a new policy instance with the same configuration as self.policy.
        Assumes that self.policy has attributes 'observation_space' and 'action_space'.
        """
        return self.policy.__class__(
            self.policy.observation_space,
            self.policy.action_space,
            lr_schedule=lambda _: self.learning_rate,  # Wrap learning_rate in a lambda
            **self.policy_kwargs
        )
    def get_log_probs(self, policy):
        """
        Compute log probabilities of actions under a given policy.
        Uses a batch from the rollout buffer.
        """
        with torch.no_grad():
            # Get one batch from the rollout buffer (this is a workaround to compute log probabilities)
            data = next(iter(self.rollout_buffer.get(self.batch_size)))
            actions = data.actions if not isinstance(self.action_space, spaces.Discrete) else data.actions.long().flatten()
            # Evaluate actions; index [1] returns log probabilities as per the PPO evaluate_actions API
            log_probs = policy.evaluate_actions(data.observations, actions)[1]
        return log_probs