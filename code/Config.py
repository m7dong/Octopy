import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--num_gpu', type=int,
                                 default=2, help='Number of GPUs')
        self.parser.add_argument('--num_local_models_per_gpu', type=int,
                                 default=5, help='Number of local models per GPU')
        self.parser.add_argument('--num_users', type=int,
                                 default=10, help='Total number of users')
        self.parser.add_argument('--num_steps', type=int,
                                 default=5, help='Number of steps training federated model')
        self.parser.add_argument('--num_epochs', type=int,
                                 default=1, help='Number of epochs training local models')
        self.parser.add_argument('--optimizer', type=str,
                                 default='SGD', help='optimizer type: one of SGD | Adam ')
        self.parser.add_argument('--lr', type=int,
                                 default=0.001, help='learning rate')
        self.parser.add_argument('--fed_strategy', type=str,
                                 default='Avg', help='fed_strategy: Avg')
        self.parser.add_argument('--loss_func', type=str,
                                 default='nll', help='loss function: one of NLL')

    def parse_args(self):
        return self.parser.parse_args()
