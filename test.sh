LOSS='mle'
seed=1
lam=0.0
censor='true'
interpolate='true'
NAME="CAT_interp${interpolate}_loss${LOSS}_seed${seed}_lam${lam}_censor${censor}"

pythonscript=" python test.py
--data_dir data
--model_dist cat
--num_cat_bins 50
--dataset synthetic
--synthetic_dist gamma
--name ${NAME}
--model GammaNN
--batch_size 1000
--phase test
--pred_type mode
--seed ${seed}
--lam ${lam}
--dropout_rate 0.0
--censor ${censor}
--interpolate ${interpolate}
--ckpt_path ckpts/${NAME}dssyntheticlam${lam}dr0.0_bs1000/best.pth.tar
"
                
runcommand=${pythonscript}
echo $runcommand
if [ "$1" = "r" ]; then
    eval $runcommand
fi
