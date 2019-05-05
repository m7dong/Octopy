import argparse
import models
class Config():
    def __init__(self):
        model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--num_gpu', type=int,
                                 default=2, help='Number of GPUs')
        self.parser.add_argument('--num_local_models_per_gpu', type=int,
                                 default=2, help='Number of local models per GPU')
        self.parser.add_argument('--num_users', type=int,
                                 default=600, help='Total number of users')
        self.parser.add_argument('--num_steps', type=int,
                                 default=1000, help='Number of steps training federated model')

        self.user_config = self.parser.add_argument_group('user_group')
        self.user_config.add_argument('--local_epoch', type=int,
                                 default=1, help='Number of epochs training local models')
        self.user_config.add_argument('--optimizer', type=str,
                                 default='SGD', help='optimizer type: one of SGD | Adam ')
        self.user_config.add_argument('--lr', type=int,
                                 default=0.001, help='learning rate')
        self.user_config.add_argument('--loss_func', type=str,
                                 default='nll', help='loss function: one of NLL')
        self.user_config.add_argument('--local_batchsize', type=int,
                                 default='50', help='local_batchsize')


        self.parser.add_argument('--fed_strategy', type=str,
                                 default='Avg', help='fed_strategy: Avg')
        self.parser.add_argument('--model', metavar='MODEL', default='LeNet',
                                choices=model_names, help='model architecture: ' +
                                ' | '.join(model_names))

    def parse_args(self):
        return self.parser.parse_args('')
