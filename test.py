import torch
import torch.nn.functional as F
import torch.utils.data as data
import util
import random
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from args import TestArgParser
from logger import TestLogger
from saver import ModelSaver
import numpy as np
from evaluator import ModelEvaluator
import warnings
import pickle
import util

def log_normal_mode(pytorch_distirbution_object):
    return (pytorch_distirbution_object.loc - pytorch_distirbution_object.scale.pow(2)).exp()

def test(args):

    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args)
    args.start_epoch = ckpt_info['epoch'] + 1
    args.device = DEVICE
    model = model.to(args.device)
    model.eval()

    train_loader = util.get_train_loader(args,during_training=False)     
    eval_loaders = util.get_eval_loaders(during_training=False, args=args)
    valid_loader, test_loader = eval_loaders
    
    # Calculate Log_likelihood
    args.loss_fn = 'mle'

    evaluator = ModelEvaluator(args, [train_loader,valid_loader,test_loader])
    
    metrics = evaluator.evaluate(model, DEVICE)    
    
    metrics['test_concordance'] = util.concordance(args,test_loader, model)
    metrics['test_set_size'] = len(test_loader.dataset)
    metrics['train_set_size'] = len(train_loader.dataset)
    metrics['val_set_size'] = len(valid_loader.dataset)
    
    return metrics

if __name__ == '__main__':
    parser = TestArgParser()
    args = parser.parse_args()

    if args.dataset=='mnist':
        assert args.model == 'SurvMNISTNN', "if dataset == mnist, model must be SurvMNISTNN"

    # Get min and max for time
    if args.model_dist in ['cat', 'mtlr']:
        bin_boundaries, mid_points, marginal_counts = util.get_bin_boundaries(args)
        args.bin_boundaries = bin_boundaries
        args.mid_points = mid_points
        args.marginal_counts = marginal_counts

    with torch.no_grad():
        metrics = test(args)

    for k in metrics:
        obj = metrics[k]
        if torch.is_tensor(obj):
            metrics[k] = obj.cpu().numpy()
        print(k, metrics[k])

    save_name = args.results_dir+args.name+'ds'+args.dataset+'lam'+str(args.lam)+'dr'+str(args.dropout_rate) + '_bs' + str(args.batch_size)
    f = open(save_name+".pkl", "wb")
    pickle.dump(metrics, f)
    print("Wrote results to:",save_name+".pkl")
    f.close()
    sys.stdout.flush()
