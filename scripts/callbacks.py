from stable_baselines3.common.callbacks import EvalCallback
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class EvalCallbackExtended(EvalCallback):
    """
    Works with older SB3 where evaluations_results = [mean_r, std_r, mean_len, std_len].
    Logs mean reward, std, ep length, std into TensorBoard.
    """

    def __init__(self, *args, writer: SummaryWriter = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.writer is None or len(self.evaluations_results) == 0:
            return result

        # Latest evaluation (format: [mean_reward, std_reward, mean_length, std_length])
        last_eval = self.evaluations_results[-1]

        mean_reward = last_eval[0]
        std_reward = last_eval[1]
        mean_length = last_eval[2]
        std_length = last_eval[3]

        # SB3 stores the timestep for each evaluation
        timestep = self.evaluations_timesteps[-1]

        self.writer.add_scalar("eval/mean_reward", mean_reward, timestep)
        self.writer.add_scalar("eval/std_reward", std_reward, timestep)
        self.writer.add_scalar("eval/mean_ep_length", mean_length, timestep)
        self.writer.add_scalar("eval/std_ep_length", std_length, timestep)

        return result
