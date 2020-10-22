import models
import os
import shutil
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ModelSaver(object):
    """Class to save and load model ckpts."""
    def __init__(self, save_dir, epochs_per_save, max_ckpts, metric_name='valid_loss', maximize_metric=False, **kwargs):
        """
        Args:
            save_dir: Directory to save checkpoints.
            epochs_per_save: Number of epochs between each save.
            max_ckpts: Maximum number of checkpoints to keep before overwriting old ones.
            metric_name: Name of metric used to determine best model.
            maximize_metric: If true, best checkpoint is that which maximizes the metric value passed in via save.
            If false, best checkpoint minimizes the metric.
        """
        super(ModelSaver, self).__init__()

        self.save_dir = save_dir
        self.epochs_per_save = epochs_per_save
        self.max_ckpts = max_ckpts
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_metric_val = None
        self.ckpt_paths = []

        self.args = [kwargs['batch_size'], kwargs['num_workers'], kwargs['data_dir']]

    def _is_best(self, metric_val):
        """Check whether metric_val is the best one we've seen so far."""
        if metric_val is None:
            return False
        return (self.best_metric_val is None
                or (self.maximize_metric and self.best_metric_val < metric_val)
                or (not self.maximize_metric and self.best_metric_val > metric_val))

    def save(self, epoch, model, optimizer, lr_scheduler, device, metric_val):
        """If this epoch corresponds to a save epoch, save model parameters to disk.

        Args:
            epoch: Epoch to stamp on the checkpoint.
            model: Model to save.
            optimizer: Optimizer for model parameters.
            lr_scheduler: Learning rate scheduler for optimizer.
            device: Device where the model/optimizer parameters belong.
            metric_val: Value for determining whether checkpoint is best so far.
        """


        import pdb
        #pdb.set_trace()
        #print(f'batch size: {self.args[0]}, num workers: {self.args[1]}')

        if epoch % self.epochs_per_save != 0:
            return

        ckpt_dict = {
            'ckpt_info': {'epoch': epoch, self.metric_name: metric_val},
            #'model_name': model.module.__class__.__name__,
            'model_name': model.kwargs['model'],
            #'model_args': model.module.args_dict(),
            'model_args': {'D_in' : model.D_in},
            'model_state': model.to('cpu').state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }
        model.to(device)

        ckpt_path = os.path.join(self.save_dir, 'epoch_{}.pth.tar'.format(epoch))
        torch.save(ckpt_dict, ckpt_path)

        if self._is_best(metric_val):
            # Save the best model
            self.best_metric_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(ckpt_path, best_path)

        # Remove a checkpoint if more than max_ckpts ckpts saved
        self.ckpt_paths.append(ckpt_path)
        if len(self.ckpt_paths) > self.max_ckpts:
            oldest_ckpt = self.ckpt_paths.pop(0)
            os.remove(oldest_ckpt)

    @classmethod
    def load_model(cls, ckpt_path, args):
        """Load model parameters from disk.

        Args:
            ckpt_path: Path to checkpoint to load.
            gpu_ids: GPU IDs for DataParallel.

        Returns:
            Model loaded from checkpoint, dict of additional checkpoint info (e.g. epoch, metric).
        """
        ckpt_dict = torch.load(ckpt_path, map_location=DEVICE)

        # Build model, load parameters
        model_fn = models.__dict__[ckpt_dict['model_name']]
        model_args = ckpt_dict['model_args']
        if not model_args:
            model_args = {'D_in' : 6800}
        # Manually loaded from test bash file
        model_args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args['dropout_rate'] = args.dropout_rate
        model_args['model_dist'] = args.model_dist
        model_args['num_cat_bins'] = args.num_cat_bins
        model_args['use_max_pool'] = args.use_max_pool
        model_args['big_gamma_nn'] = args.big_gamma_nn
        model_args['hiddensize'] = args.hiddensize
        model_args['lognormal_layers'] = args.lognormal_layers
        model = model_fn(data_dir="data", **model_args)
        #model = nn.DataParallel(model, gpu_ids)
        model.load_state_dict(ckpt_dict['model_state'])

        return model, ckpt_dict['ckpt_info']

    @classmethod
    def load_optimizer(cls, ckpt_path, optimizer, lr_scheduler=None):
        """Load optimizer and LR scheduler state from disk.

        Args:
            ckpt_path: Path to checkpoint to load.
            optimizer: Optimizer to initialize with parameters from the checkpoint.
            lr_scheduler: Optional learning rate scheduler to initialize with parameters from the checkpoint.
        """
        ckpt_dict = torch.load(ckpt_path, map_location=DEVICE)
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
