import os
import datetime
from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari
# import model
from config import *
from model.cnn_to_mlp import cnn_to_mlp
from model.cnn_to_lstm import cnn_to_lstm
from model.mlp import mlp
from model.lstm_to_mlp import lstm_to_mlp
from model.cnn_to_lstm_new import cnn_to_lstm_new

def main():

    model_choices = ["atari_deepmind",
                     "cnn_to_lstm",
                     "mlp",
                     "lstm_to_mlp",
                     "cnn_to_lstm_new"]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", help="environment ID", default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    parser.add_argument("--prioritized", type=int, default=1)
    parser.add_argument("--dueling", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=int(5*10e2))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=int(1e6))
    parser.add_argument("--exploration_steps", type=float, default=1e6)
    parser.add_argument("--exploration_final_eps", type=float, default=0.1)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--learning-starts", type=int, default=int(1e4))
    parser.add_argument("--target_network_update_freq", type=int, default=int(1e3))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--model", type=str, choices = model_choices, default="cnn_to_lstm_new")
    args = parser.parse_args()

    logger.configure(log_dir)

    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)

    if args.model == "mlp":
        model = mlp(hiddens=[256, 256])
    elif args.model == "atari_deepmind":
        model = cnn_to_mlp(convs=[(16, 8, 4), (32, 4, 2)],
                                                hiddens=[256],
                                                duelings=bool(args.dueling))
    elif args.model == "cnn_to_lstm":
        model = cnn_to_lstm(convs=[(16, 8, 4), (32, 4, 2)],
                            lstm_hidden_size=512,
                            lstm_out_size=256,
                            hiddens=[256, 128],
                            batch_size=int(args.batch_size),
                            duelings=bool(args.dueling))
    elif args.model == "cnn_to_lstm_new":
        model = cnn_to_lstm_new(convs=[(16, 8, 4), (32, 4, 2)],
                            lstm_hidden_size=512,
                            lstm_out_size=256,
                            hiddens=[256, 128],
                            batch_size=int(args.batch_size),
                            duelings=bool(args.dueling))
    elif args.model == "lstm_to_mlp":
        model = lstm_to_mlp(lstm_hidden_size=512,
                            lstm_out_size=256,
                            hiddens=[256, 128],
                            batch_size=int(args.batch_size),
                            duelings=bool(args.dueling))

    act = deepq.learn(
        env,
        q_func=model,
        lr=args.learning_rate,
        max_timesteps=args.num_timesteps,
        buffer_size=int(args.buffer_size),
        exploration_fraction=(args.exploration_steps / args.num_timesteps),
        exploration_final_eps=args.exploration_final_eps,
        train_freq=args.train_freq,
        batch_size=int(args.batch_size),
        learning_starts=int(args.learning_rate),
        target_network_update_freq=int(args.target_network_update_freq),
        gamma=args.gamma,
        prioritized_replay=bool(args.prioritized)
    )

    f = open(os.path.join(log_dir, "README.me"), "w")
    f.write("\tenv \t{}\n".format(args.env))
    f.write("\tmodel\t{}\n".format(args.model))
    f.write("\tprioritized\t{}\n".format(args.prioritized))
    f.write("\tdueling\t{}\n".format(args.dueling))
    f.write(("\tlearning rate\t{}\n".format(args.learning_rate)))
    f.write(("\tbatch size\t{}\n").format(args.batch_size))
    f.write("\tmax timestep\t{}\n".format(args.num_timesteps))
    f.write("\tbuffer size\t{}\n".format(args.buffer_size))
    f.write("\texploration fraction\t{}\n".format(args.exploration_steps/args.num_timesteps))
    f.write("\texploration_final_eps\t{}\n".format(args.exploration_final_eps))
    f.write("\ttrain freq\t{}\n".format(args.train_freq))
    f.write("\tlearning starts\t{}\n".format(args.learning_rate))
    f.write("\ttarget network update freq\t{}\n".format(args.target_network_update_freq))
    f.close()
    act.save("log/{}.pkl".format(args.model + "_" + args.env.replace(" ", "_")))

if __name__ == '__main__':
    main()