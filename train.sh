LOSS='mle'
seed=1
lam=0.0
censor='true'
interpolate='true'
NAME="CAT_interp${interpolate}_loss${LOSS}_seed${seed}_lam${lam}_censor${censor}"

pythonscript=" python train.py
--train_gamma 10000
--data_dir data
--model_dist cat
--num_cat_bins 50
--dataset synthetic
--synthetic_dist gamma
--name ${NAME}
--num_epochs 500
--loss_fn ${LOSS}
--model GammaNN
--batch_size 1000
--lr 1e-3
--epochs_per_save 1
--optimizer adam
--pred_type mode
--seed ${seed}
--lam ${lam}
--dropout_rate 0.0
--censor ${censor}
--interpolate ${interpolate}
--iters_per_print 10000"
                
runcommand=${pythonscript}
echo $runcommand
if [ "$1" = "r" ]; then
    eval $runcommand
fi
