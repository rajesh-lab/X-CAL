import util

from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False
        self.parser.add_argument('--results_dir', type=str, default='results/', help='Save dir for test results.')

        self.parser.add_argument('--ntrials', type=int, default=1, help='Save dir for test results.')

        self.parser.add_argument('--test_gamma', type=int, default=100000., help='Save dir for test results.')
        self.parser.add_argument('--threshold', type=float, default=0.5, help='treshold for classification')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4,
                                                 help='Weight decay (i.e., L2 regularization factor).')
        self.parser.add_argument('--experiment_number', type=int, default = -1)

