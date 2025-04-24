import torch
import torch.nn as nn
import numpy as np
from llama_cpu import Transformer, ModelArgs  # On suppose que ce module est disponible

# Ajout imports pour Gym et Stable Baselines3
import gym
from gym import spaces
from stable_baselines3 import PPO


class StudentTeacherEnv(gym.Env):
    """
    Environnement Gym pour RL où l'agent contrôle les poids du student.
    L'action est un incrément à appliquer aux poids du student.
    L'observation est la différence de poids (ou les poids du student).
    La reward est (theta_teacher-theta_student_old)^2 + (theta_teacher-theta_student_new)^2.
    L'épisode se termine si la MSE < eps.
    """

    def __init__(self, teacher, student, eps=1e-3):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.eps = eps
        # On vectorise tous les poids dans un seul vecteur pour l'action et l'observation
        self.teacher_params = self._get_flat_params(self.teacher)
        self.student_params = self._get_flat_params(self.student)
        param_dim = self.teacher_params.shape[0]
        # Action: incrément à appliquer aux poids du student
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(param_dim,), dtype=np.float32
        )
        # Observation: différence entre teacher et student
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(param_dim,), dtype=np.float32
        )
        self.reset()

    def _get_flat_params(self, model):
        params = []
        for p in model.parameters():
            params.append(p.detach().cpu().numpy().flatten())
        return np.concatenate(params)

    def _set_flat_params(self, model, flat_params):
        idx = 0
        for p in model.parameters():
            numel = p.numel()
            new_data = (
                torch.from_numpy(flat_params[idx : idx + numel]).view_as(p).to(p.device)
            )
            p.data.copy_(new_data)
            idx += numel

    def reset(self):
        # Réinitialise le student à des poids aléatoires
        for p in self.student.parameters():
            nn.init.normal_(p, mean=0, std=0.02)
        self.student_params = self._get_flat_params(self.student)
        self.teacher_params = self._get_flat_params(self.teacher)
        obs = self.teacher_params - self.student_params
        self.prev_mse = np.mean((self.teacher_params - self.student_params) ** 2)
        return obs.astype(np.float32)

    def step(self, action):
        # Applique l'action (incrément sur les poids du student)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        student_params_old = self.student_params.copy()
        new_params = self.student_params + action
        self._set_flat_params(self.student, new_params)
        self.student_params = self._get_flat_params(self.student)
        mse_old = np.mean((self.teacher_params - student_params_old) ** 2)
        mse_new = np.mean((self.teacher_params - self.student_params) ** 2)
        reward = -(mse_old + mse_new)
        done = mse_new < self.eps
        obs = self.teacher_params - self.student_params
        info = {"mse": mse_new}
        return obs.astype(np.float32), reward, done, info


# =============================================================================
# Instanciation des trois modèles
# =============================================================================
vocab_size = 10
params = ModelArgs(16, 2, 2, vocab_size=vocab_size)

teacher = Transformer(params)
student = Transformer(params)
for param in teacher.parameters():
    param.requires_grad = False

# Création de l'environnement Gym
env = StudentTeacherEnv(teacher, student, eps=1e-3)

# Stable Baselines3 PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

print("Entraînement terminé.")
