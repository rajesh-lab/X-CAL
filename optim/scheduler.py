import torch.optim as optim


def get_scheduler(optimizer, args):
    """Get a learning rate scheduler.

    Args:
        optimizer: The optimizer whose learning rate is modified by the returned scheduler.
        args: Command line arguments.

    Returns:
        PyTorch scheduler that update the learning rate for `optimizer`.
    """
    if args.lr_scheduler == 'step' or args.lr_scheduler == 'none':
        # if none, we wont step it in training
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler == 'multi_step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=0.5,
                                                        #  args.lr_decay_gamma,
                                                         min_lr=0.01,
                                                         patience=args.lr_patience)
    else:
        raise ValueError('Invalid learning rate scheduler: {}.'.format(args.lr_scheduler))

    return scheduler


def step_scheduler(lr_scheduler, metrics, epoch, best_ckpt_metric='valid_loss'):
    """Step a LR scheduler."""
    if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if best_ckpt_metric in metrics:
            print("TRYING TO STEP PLATEAU")
            lr_scheduler.step(metrics[best_ckpt_metric], epoch=epoch)
    else:
        lr_scheduler.step(epoch=epoch)
