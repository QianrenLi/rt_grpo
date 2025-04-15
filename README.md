# rt_grpo

The training part relies on the `rl-baselines3-zoo`, which serves as the backbone.

## Setup
To set up the environment, you can follow the instructions in the [`rl-baselines3-zoo`](https://github.com/DLR-RM/rl-baselines3-zoo/tree/506bb7aa40e9d90e997580a369f2e9bf64abe594) repository.

## Training
To run the GRPO algorithm, you need to first use the `backbone_setup.py` script to set up the backbone correctly.
After that, the `train.py` script can be used to train the baseline model (e.g. PPO A2C) as a initial point for GRPO.
With the trained model, you can then run the `GRPO.py` script to train the GRPO model.
An example of the training process is
```bash
python backbone_setup.py
python3 GRPO.py  --env MISOEnv-antenna-4 --exp-id 1 --algo ppo --folder ./logs/
```
where `--env` specifies the environment, `--exp-id` specifies the experiment ID, `--algo` specifies the algorithm in the initial point  (e.g. PPO), and `--folder` specifies the folder where the logs will be saved.

## Evaluation
To evaluate the trained model, you can use the `GRPO.py` script. The script takes the trained model and evaluates it on the environment. (Note the model name should be specified if you want to evaluate a specific model)

```bash
python3 GRPO.py --env MISOEnv-antenna-4 --exp-id 1 --algo ppo --folder ./logs/ --eval
```
where `--env` specifies the environment, `--exp-id` specifies the experiment ID, `--algo` specifies the algorithm in the initial point (e.g. PPO), and `--folder` specifies the folder where the logs will be saved. The `--eval` flag indicates that you want to evaluate the model.
