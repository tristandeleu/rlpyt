
class BaseRunner:
    """
    Orchestrates sampler and algorithm to run the training loop.  The runner
    should also manage logging to record agent performance during training.
    Different runner classes may be used depending on the overall RL procedure
    and the hardware configuration (e.g. multi-GPU).
    """
    _eval = False

    def __init__(self,
                 algo,
                 agent,
                 sampler,
                 n_steps,
                 affinity=None,
                 seed=None,
                 log_interval_steps=int(1e5)):
        self.algo = algo
        self.agent = agent
        self.sampler = sampler
        self.n_steps = int(n_steps)
        self.affinity = affinity or dict()
        self.seed = seed
        self.log_interval_steps = int(log_interval_steps)

    def train(self):
        """
        Entry point to conduct an entire RL training run, to be called in a
        launch script after instantiating all components: algo, agent, sampler.
        """
        raise NotImplementedError
