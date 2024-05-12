import argparse
import random

def seed(s):
    if isinstance(s, int):
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError(
                "Seed must be between 0 and 2**32 - 1. Received {0}".format(s)
            )
    elif s == "random":
        return random.randint(0, 9999)
    else:
        raise argparse.ArgumentTypeError(
            "Integer value is expected. Recieved {0}".format(s)
        )


parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis')
parser.add_argument('--model', type=str, default="chinese-xlnet-base", help='xlnet-base-cased/chinese-xlnet-base  -- model name')
parser.add_argument('--dataset', type=str, default='mosi', help='default: mosei/mosi/sims_39')
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--seed", type=seed, default=6820, help='random')
parser.add_argument("--learning_rate", type=float, default=1e-5, help='1e-5')
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--Use_EFusion", type=bool, default=False)
parser.add_argument("--Use_LFusion", type=bool, default=False)
parser.add_argument("--drop", type=float, default=0.1)
parser.add_argument("--scaling_factor", type=float, default=0.5)

args = parser.parse_args()  # 使用 parse_args() 解析添加的参数

if args.dataset == 'mosi':
    ACOUSTIC = 74
    VISUAL = 47
    TEXT = 768
elif args.dataset == 'mosei':
    ACOUSTIC = 74
    VISUAL = 35
    TEXT = 768


