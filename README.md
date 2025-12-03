## Training

We provide three training scripts, one for each method:
- `scripts/train_rmappo.py` – rMAPPO baseline  
- `scripts/train_maddpg.py` – MADDPG baseline  
- `scripts/train_epigraph.py` – Epigraph-form Safe MARL

A typical training command looks like:

```bash
python scripts/train_rmappo.py \
  --num_envs 48 \
  --max_global_steps 150000 \
  --wandb

--num_envs sets the number of parallel Isaac environments,
--max_global_steps controls the total number of environment steps,
and --wandb enables logging to Weights & Biases.

You can replace train_rmappo.py with train_maddpg.py or train_epigraph.py to train the respective method.


## Evaluation

Trained policies are evaluated with the `scripts/play_epigraph.py` script.  
A typical evaluation command is:

```bash
python scripts/play_epigraph.py \
  --checkpoint /home/zzh/workspace/shared_control_isaac_sim/logs/epigraph/20251119_192247/checkpoints/ckpt_milestone_002400_score_119.904634.pth \
  --num_episodes 1 \
  --num_envs 1 \
  --deterministic

--checkpoint specifies the path to the saved model,
--num_episodes is the number of evaluation rollouts,
--num_envs sets the number of parallel environments during evaluation,
and --deterministic disables exploration noise for a deterministic policy rollout.

You can replace play_epigraph.py with play_maddpg.py or play_rmappo.py to evaluate the respective method.