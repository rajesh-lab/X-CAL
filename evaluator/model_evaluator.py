from __future__ import print_function

import torch
import torch.nn as nn
from tqdm import tqdm

import optim
import util
import pdb
import numpy as np
from lifelines.utils import concordance_index

from evaluator.average_meter import AverageMeter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelEvaluator(object):
    """Class for evaluating a model during training."""

    def __init__(self, args, data_loaders,epochs_per_eval=1):
        """
        Args:
            data_loaders: List of Torch `DataLoader`s to sample from.
            num_visuals: Number of visuals to display from the validation set.
            max_eval: Maximum number of examples to evaluate at each evaluation.
            epochs_per_eval: Number of epochs between each evaluation.
        """
        self.args = args
        self.dataset = args.dataset
        self.data_loaders = data_loaders
        self.epochs_per_eval = epochs_per_eval
        self.loss_fn = optim.get_loss_fn(args.loss_fn, args)
        self.name = args.name
        self.lam = args.lam
        self.pred_type = args.pred_type
        self.model_dist = args.model_dist
        self.num_cat_bins = args.num_cat_bins
        self.loss_fn_name = args.loss_fn
        self.num_dcal_bins = args.num_dcal_bins
        if self.model_dist in ['cat','mtlr']:
            self.mid_points = args.mid_points
            self.bin_boundaries = args.bin_boundaries
    def evaluate(self, model, device, epoch=None):
        """Evaluate a model at the end of the given epoch.

        Args:
            model: Model to evaluate.
            device: Device on which to evaluate the model.
            epoch: The epoch that just finished. Determines whether to evaluate the model.

        Returns:
            metrics: Dictionary of metrics for the current model.

        Notes:
            Returned dictionary will be empty if not an evaluation epoch.
        """
        metrics = {}

        if epoch is None or epoch % self.epochs_per_eval == 0:
            # Evaluate on the training and validation sets
            model.eval()
            for data_loader in self.data_loaders:
                phase_metrics = self._eval_phase(model, data_loader, data_loader.phase, device)
                metrics.update(phase_metrics)
            model.train()

        return metrics

    def _eval_phase(self, model, data_loader, phase, device):
        print("CURRENT EVAL PHASE IS ", phase)


        """Evaluate a model for a single phase.

        Args:
            model: Model to evaluate.
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.

        Returns:
            metrics: Dictionary of metrics for the phase.
        """

        # Keep track of task-specific records needed for computing overall metrics
        records = {'loss_meter': AverageMeter()}

        num_examples = len(data_loader.dataset)
        # Sample from the data loader and record model outputs
        loss_fn = self.loss_fn
        num_evaluated = 0

        is_alive_per_batch = []
        
        all_cdf = []
        for src, tgt in data_loader:
            src=src.to(DEVICE)
            tgt=tgt.to(DEVICE)

            tte = tgt[:,0]


           
            
            if self.model_dist in ['cat', 'mtlr']:
                tgt = util.cat_bin_target(self.args,tgt, self.bin_boundaries)            

            # THIS MUST COME AFTER THE ABOVE CAT BIN TARGET FUNCTION
            # BECAUSE CAT BIN TARGET CAN CHANGE SOME IS_ALIVE
            is_alive =tgt[:,1]
            
            
            is_alive_per_batch.append(is_alive)

            if num_evaluated >= num_examples:
                break


            pred_params = model.forward(src)
            cdf = util.get_cdf_val(pred_params, tgt, self.args)
            all_cdf.append(cdf)
            loss = loss_fn(pred_params, tgt, model_dist=self.model_dist) 

            num_both = pred_params.size()[0]

            self._record_batch(num_both, loss, **records)
            num_evaluated += src.size(0)


        if self.args.dataset in ['nacd','nacdcol','brca','read','gbm','gbmlgg','dbcd','dlbcl']:
            concordance = util.concordance(self.args,data_loader, model)
        else:
            concordance = -1.0

 
        is_alive = torch.cat(is_alive_per_batch).long()

        all_cdf = torch.cat(all_cdf)
  


        # Map to summary dictionaries
        metrics = self._get_summary_dict(phase, **records)

        
        d_calibration = util.d_calibration(points=all_cdf, args=self.args, nbins=self.num_dcal_bins, is_alive=is_alive)
        
        test_statistic,p_value = util.get_p_value(d_cal = d_calibration,
                                   degree_of_freedom = self.num_dcal_bins-1,
                                   num_of_samples = num_examples,
                                   num_of_bins = self.num_dcal_bins)
        
        approx_d_calibration = util.d_calibration(points=all_cdf, is_alive=is_alive, args=self.args,nbins=self.num_dcal_bins,
                                                  differentiable=True,gamma=1e5, device=DEVICE)

        '''
        approx_d_calibration_10 = util.d_calibration(points=all_cdf, is_alive=is_alive, args=self.args,nbins=self.num_dcal_bins,
                                                     differentiable=True,
                                                     gamma=1e6, device=DEVICE)
        '''


        metrics[phase + '_' + 'NLL'] = metrics[phase + '_' + 'loss']
        metrics[phase + '_' + 'dcal'] = d_calibration
        metrics[phase + '_' + 'approxdcal'] = approx_d_calibration
        metrics[phase + '_' + 'loss'] = metrics[phase + '_' + 'loss'] + self.lam * d_calibration
        metrics[phase + '_' + 'teststat'] = test_statistic
        metrics[phase + '_' + 'pvalue'] = p_value
        metrics[phase + "_" + 'concordance'] = concordance

        print(' ---- {} epoch end D-cal {:.3f}, approx {:.3f}'.format(phase, d_calibration, approx_d_calibration))
        print(' ---- {} epoch Concordance {:.3f}'.format(phase, concordance))
        return metrics

    @staticmethod
    def _record_batch(N, loss, loss_meter=None):
        """Record results from a batch to keep track of metrics during evaluation.

        Args:
            logits: Batch of logits output by the model.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        """
        if loss_meter is not None:
            loss_meter.update(loss.item(), N)

    @staticmethod
    def _get_summary_dict(phase, loss_meter=None):
        """Get summary dictionaries given dictionary of records kept during evaluation.

        Args:
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            loss_meter: AverageMeter keeping track of average loss during evaluation.

        Returns:
            metrics: Dictionary of metrics for the current model.
        """
        metrics = {phase + '_' + 'loss': loss_meter.avg}

        return metrics

    @staticmethod
    def _write_summary_stats(phase, loss_meter=None):
        """Write stats of evaluation to file.

        Args:
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            loss_meter: AverageMeter keeping track of average loss during evaluation.

        Returns:
            metrics: Dictionary of metrics for the current model.
        """
        metrics = {phase + '_' + 'loss': loss_meter.avg}

        return metrics
