import torch
from configs import args
from src import train
from src.utils import set_up_data_loader, set_random_seed


set_random_seed(args.seed)

torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_loader, valid_loader, test_loader, num_train_optimization_steps = set_up_data_loader(args)

hyp_params = args
hyp_params.device = device
hyp_params.name = args.model + str(args.Use_EFusion) + str(args.Use_LFusion) + str(args.Use_Mag)


if __name__ == '__main__':
    train.train(hyp_params, train_loader, valid_loader, test_loader, num_train_optimization_steps)

