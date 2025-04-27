import numpy as np
import torch
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3 import PPO
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from llama_cpu import Transformer, ModelArgs


# =============================================================================
# 2. Custom feature extractor using llama_cpu.Transformer without causal mask
# =============================================================================
class LlamaExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, seq_len: int, embed_dim: int):
        super().__init__(observation_space, features_dim=observation_space.shape[0])
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        # use llama_cpu Transformer as feature extractor (no causal mask)
        params = ModelArgs(
            embed_dim, 2, 2, vocab_size=1, apply_tok_embeddings=False
        )
        params.causal_mask = False
        self.transformer = Transformer(params)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: [batch, 2*seq_len, embed_dim]
        # we reinterpret embeddings as token indices by projecting to vocab space
        # here, a simple linear layer to map embeddings to logits over vocab (dummy)
        # but we can directly call transformer on embedded inputs
        # Pass observations as if they were token embeddings
        # transformer in llama_cpu will skip embedding layer if given embeddings
        out = self.transformer(observations[0], 0)  # [batch, seq, embed_dim]
        # mean-pool over sequence dimension
        pooled = out.mean(dim=(1, 2)).unsqueeze(0)  # [1, batch]
        return pooled


# =============================================================================
# 3. Custom policy (LossNetwork) using LlamaExtractor
# =============================================================================
class LossNetwork(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=LlamaExtractor,
            features_extractor_kwargs={
                "seq_len": kwargs.pop("seq_len"),
                "embed_dim": kwargs.pop("embed_dim"),
            },
            **kwargs,
        )


# =============================================================================
# 4. Environment definition
# =============================================================================
class LossEnv(Env):
    def __init__(
        self,
        seq_len: int = 32,
        embed_dim: int = 16,
        vocab_size: int = 1000,
        batch_size: int = 4,
        max_steps: int = 20,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.current_step = 0
        self.eps = eps

        # obs: teacher_out + student_out -> [2*seq_len, embed_dim]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(batch_size, 2 * self.seq_len, self.embed_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0, high=1000.0, shape=(batch_size,), dtype=np.float32)
        self.params = ModelArgs(self.embed_dim, 2, 2, vocab_size=vocab_size)
        self.params.causal_mask = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.teacher = Transformer(self.params)
        self.student = Transformer(self.params)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.student_optimizer = optim.AdamW(self.student.parameters(), lr=1e-3)
        self.current_step = 0
        X = torch.randint(0, self.params.vocab_size, (self.batch_size, self.seq_len))
        teacher_out = self.teacher(X, 0)
        student_out = self.student(X, 0)
        obs = torch.cat([teacher_out, student_out], dim=1)
        info = {}
        return obs.cpu().numpy(), info

    def step(self, action):
        loss_value = torch.tensor(action)
        # nouveau batch
        X = torch.randint(0, self.params.vocab_size, (self.batch_size, self.seq_len))
        # compute weight mse before
        mse_before = sum(
            torch.mean((pt - ps) ** 2)
            for pt, ps in zip(self.teacher.parameters(), self.student.parameters())
        )

        # on freeze les poids du LossNetwork

        # update student with loss network output
        self.student_optimizer.zero_grad()
        # student_loss = (loss_value * mse_before).mean()
        student_loss = loss_value.mean()
        student_loss.backward()
        self.student_optimizer.step()

        # on défreeze les poids du LossNetwork

        # compute weight mse after
        mse_after = sum(
            torch.mean((pt - ps) ** 2)
            for pt, ps in zip(self.teacher.parameters(), self.student.parameters())
        )
        reward = (mse_before - mse_after).item()
        # new observation (no no_grad to allow gradient elsewhere)
        teacher_out = self.teacher(X, 0)
        student_out = self.student(X, 0)
        obs = torch.cat([teacher_out, student_out], dim=1)
        self.current_step += 1
        done = bool(self.current_step >= self.max_steps or mse_after < self.eps)
        return obs.cpu().numpy(), reward, done, False, {}


# =============================================================================
# 5. Training
# =============================================================================
if __name__ == "__main__":
    env = LossEnv(seq_len=32, embed_dim=16, vocab_size=100, batch_size=4, max_steps=20)
    check_env(env)

    model = PPO(
        policy=LossNetwork,
        env=env,
        verbose=1,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        tensorboard_log="./ppo_lossnet_policy/",
        policy_kwargs={"seq_len": 32, "embed_dim": 16},
    )
    model.learn(total_timesteps=200_000)
    model.save("ppo_lossnet_llama_policy")
    print("Training complete.")
