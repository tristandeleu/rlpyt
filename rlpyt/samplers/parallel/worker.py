
import psutil
import time
import torch

from rlpyt.utils.logging import logger
from rlpyt.utils.seed import set_seed, set_envs_seeds


def initialize_worker(rank, seed=None, cpu=None, torch_threads=None):
    """Assign CPU affinity, set random seed, set torch_threads if needed to
    prevent MKL deadlock.
    """
    log_str = f"Sampler rank {rank} initialized"
    cpu = [cpu] if isinstance(cpu, int) else cpu
    p = psutil.Process()
    try:
        if cpu is not None:
            p.cpu_affinity(cpu)
        cpu_affin = p.cpu_affinity()
    except AttributeError:
        cpu_affin = "UNAVAILABLE MacOS"
    log_str += f", CPU affinity {cpu_affin}"
    torch_threads = (1 if torch_threads is None and cpu is not None else
        torch_threads)  # Default to 1 to avoid possible MKL hang.
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    log_str += f", Torch threads {torch.get_num_threads()}"
    if seed is not None:
        set_seed(seed)
        time.sleep(0.3)  # (so the printing from set_seed is not intermixed)
        log_str += f", Seed {seed}"
    logger.log(log_str)


def sampling_process(common_kwargs, worker_kwargs):
    """Target function used for forking parallel worker processes in the
    samplers. After ``initialize_worker()``, it creates the specified number
    of environment instances and gives them to the collector when
    instantiating it.  It then calls collector startup methods for
    environments and agent.  If applicable, instantiates evaluation
    environment instances and evaluation collector.

    Then enters infinite loop, waiting for signals from master to collect
    training samples or else run evaluation, until signaled to exit.
    """
    initialize_worker(worker_kwargs['rank'], worker_kwargs['seed'],
        worker_kwargs['cpus'], common_kwargs['torch_threads'])
    envs = [common_kwargs['EnvCls'](**common_kwargs['env_kwargs'])
            for _ in range(worker_kwargs['n_envs'])]
    set_envs_seeds(envs, worker_kwargs['seed'])

    collector = common_kwargs['CollectorCls'](
        rank=worker_kwargs['rank'],
        envs=envs,
        samples_np=worker_kwargs['samples_np'],
        batch_T=common_kwargs['batch_T'],
        TrajInfoCls=common_kwargs['TrajInfoCls'],
        agent=common_kwargs.get("agent", None),  # Optional depending on parallel setup.
        sync=worker_kwargs.get("sync", None),
        step_buffer_np=worker_kwargs.get("step_buffer_np", None),
        global_B=common_kwargs.get("global_B", 1),
        env_ranks=worker_kwargs.get("env_ranks", None),
    )
    agent_inputs, traj_infos = collector.start_envs(common_kwargs['max_decorrelation_steps'])
    collector.start_agent()

    if common_kwargs.get("eval_n_envs", 0) > 0:
        eval_envs = [common_kwargs['EnvCls'](**common_kwargs['eval_env_kwargs'])
                     for _ in range(common_kwargs['eval_n_envs'])]
        set_envs_seeds(eval_envs, worker_kwargs['seed'])
        eval_collector = common_kwargs['eval_CollectorCls'](
            rank=worker_kwargs['rank'],
            envs=eval_envs,
            TrajInfoCls=common_kwargs['TrajInfoCls'],
            traj_infos_queue=common_kwargs['eval_traj_infos_queue'],
            max_T=common_kwargs['eval_max_T'],
            agent=common_kwargs.get("agent", None),
            sync=worker_kwargs.get("sync", None),
            step_buffer_np=worker_kwargs.get("eval_step_buffer_np", None),
        )
    else:
        eval_envs = list()

    ctrl = common_kwargs['ctrl']
    ctrl.barrier_out.wait()
    while True:
        collector.reset_if_needed(agent_inputs)  # Outside barrier?
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            break
        if ctrl.do_eval.value:
            eval_collector.collect_evaluation(ctrl.itr.value)  # Traj_infos to queue inside.
        else:
            agent_inputs, traj_infos, completed_infos = collector.collect_batch(
                agent_inputs, traj_infos, ctrl.itr.value)
            for info in completed_infos:
                common_kwargs['traj_infos_queue'].put(info)
        ctrl.barrier_out.wait()

    for env in envs + eval_envs:
        env.close()
