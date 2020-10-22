import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import models
import optim
import util
from args import TrainArgParser
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver
import pdb
import numpy as np
import random

# computes stochastic upper bound on differentiable dcal, aka xcal
def compute_xcal(pred_params,tgt,args):

    cdf = util.get_cdf_val(pred_params, tgt, args)

    # ratio not used here, no need to check for 3rd dim
    tte, is_alive = tgt[:, 0], tgt[:, 1]


    d_cal = util.d_calibration(points=cdf, 
                               is_alive=is_alive,
                               args=args,
                               nbins=args.num_dcal_bins,
                               differentiable=True, 
                               gamma=args.train_gamma, 
                               device=DEVICE)
    
    return d_cal


def train(args):
   
    train_loader = util.get_train_loader(args)    
  

    # load a checkpoint model
    args.device = DEVICE
    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        args.D_in = train_loader.D_in
        model = model_fn(**vars(args))
    model = model.to(args.device)


    model.train()
    # Get optimizer and scheduler
    optimizer = optim.get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    lr_scheduler = optim.get_scheduler(optimizer, args)
    

    # load an optimizer checkpoint
    if args.ckpt_path:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)


    # Get logger, evaluator, saver
    loss_fn = optim.get_loss_fn(args.loss_fn, args)

    logger = TrainLogger(args, len(train_loader.dataset))


    eval_loaders = util.get_eval_loaders(during_training=True, args=args)

    evaluator = ModelEvaluator(args, eval_loaders)

    saver = ModelSaver(**vars(args))
    
    # Evaluate before training
    with torch.no_grad():
        metrics = evaluator.evaluate(model, args.device, 0)

    # Train model
    if args.lam > 0.0:
            lam  = args.lam
    else:
        lam = 0.0


    while not logger.is_finished_training():
        logger.start_epoch()
        d_cal_accumulator = 0.0
        
        print("******* STARTING TRAINING LOOP ******* ")
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('current lam', lam)
        print('current lr',cur_lr)

        for src, tgt in train_loader:
            logger.start_iter()

            src=src.to(DEVICE)
            tgt=tgt.to(DEVICE)

            with torch.set_grad_enabled(True):

                if torch.any(torch.isnan(src)):
                    print("SRC HAS NAN")
                if torch.any(torch.isnan(tgt)):
                    print("TGT HAS NAN")

                model.train()
                pred_params = model.forward(src.to(args.device))

                if args.model_dist in ['cat','mtlr']:
                    '''
                    bin the times for cat model
                    when we use interpolated cdf,
                    we also use the "ratio" also saved in target
                    to recover the continuous information
                    '''
                    tgt = util.cat_bin_target(args,tgt, bin_boundaries) 

                # compute loss 

                loss = 0
                if not args.loss_xcal_only:
                    loss += loss_fn(pred_params, tgt, model_dist=args.model_dist)

                # compute regularizers
                if args.lam > 0 or args.loss_xcal_only:
               
                    d_cal = compute_xcal(pred_params, tgt,args)

                    d_cal_accumulator  += d_cal.detach().item()

                    if args.loss_xcal_only:
                        loss = d_cal
                    else:
                        loss = loss + lam * d_cal

                logger.log_iter(src, pred_params, tgt, loss)
                optimizer.zero_grad()
                
                loss.backward()

                
                optimizer.step()
            
            logger.end_iter()


        print(" ********** CALLING EVAL *******")
        
        with torch.no_grad():
            metrics = evaluator.evaluate(model, args.device, logger.epoch)

        saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,\
                   metric_val=metrics.get(args.metric_name, None))
        logger.end_epoch(metrics=metrics)
        
        if args.lr_scheduler != 'none':
            optim.step_scheduler(lr_scheduler, metrics, logger.epoch)
            print("ATTEMPT STEPPING LEARNING RATE")

if __name__ == '__main__':

    torch.set_anomaly_enabled(True)
    parser = TrainArgParser()
    args = parser.parse_args()
    
    print("CUDA IS AVAILABLE:", torch.cuda.is_available())
    if args.model_dist in ['cat','mtlr']:
        bin_boundaries, mid_points = util.get_bin_boundaries(args)
        print('bin_boundaries', bin_boundaries)
        args.bin_boundaries = bin_boundaries
        args.mid_points = mid_points

    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print('dataset name', args.dataset)
    train(args)
