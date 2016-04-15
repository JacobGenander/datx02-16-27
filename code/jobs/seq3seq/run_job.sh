#!/bin/bash

# TODO: Check that the paths actually work in the queue. It is possible that
# one has to use absolute paths, which means that $PATH_CWD must be expanded
PATH_CWD="."
PATH_PYTHON=$(which python2)
PATH_SCRIPT="../../seq3seq/translate.py"
FILE_STDOUT="stdout"
FILE_STDERR="stderr"

# Grab mode as first argument, can be either 'train' or 'eval' 
MODE="$1"

# Arguments that are common for all GPU:s
# NOTE: The flag --train_dir is set further down in the script and is relative
# to the subfolder that is created for each process
ARGS_COMMON=( \
	--data_dir ~/data/ \
	--article_file articles_500000.txt \
	--title_file titles_500000.txt \
	--article_vocab_size 30000 \
	--title_vocab_size 20000 \
	--num_layers 3 \
	--steps_per_checkpoint 300 \
	--batch_size 60
)

# Arguments to append when $MODE is 'train'
ARGS_TRAIN=( \
	--decode false
)

# Arguments to append when $MODE is 'eval'
ARGS_EVAL=( \
	--decode true
)

# Arguments which are specific to the different GPU:s
# The python script should probably have a command line argument which can be
# given here, specifying which GPU to use.
# These arguments bust be in quotes as to not be interpreted as different
# array elements by bash, if empty strings, "", are given. No job is started
# on that GPU
ARGS_GPU_SPECIFIC[0]=""
ARGS_GPU_SPECIFIC[1]="--size 512"
ARGS_GPU_SPECIFIC[2]="--size 256"
ARGS_GPU_SPECIFIC[3]="--size 1024"

# Used to interate
let HIGHEST_INDEX=${#ARGS_GPU_SPECIFIC[@]}-1

# Removing previous termination script
echo "Removing previous killall script. . ."
rm kill_all.sh

# Iterate over the possible GPU:S
for GPU in $(seq 0 $HIGHEST_INDEX)
do
	if [ -z "${ARGS_GPU_SPECIFIC[$GPU]}" ];
	then
		echo "----No work specified for GPU_$GPU----"
	else
		PROCESS_FOLDER="$PATH_CWD/GPU_$GPU"
		if [ ! -d "$PROCESS_FOLDER" ]; then
			echo "Creating folder for process running on GPU_$GPU"
			mkdir "$PROCESS_FOLDER"
		fi
		CMDLINE_TRAIN=("$PATH_PYTHON" "$PATH_SCRIPT" --train_dir "$PROCESS_FOLDER" "${ARGS_COMMON[@]}" "${ARGS_TRAIN[@]}" "${ARGS_GPU_SPECIFIC[$GPU]}")
		#CMDLINE_TRAIN=("echo lol")
		CMDLINE_EVAL=("$PATH_PYTHON" "$PATH_SCRIPT" --train_dir "$PROCESS_FOLDER" "${ARGS_COMMON[@]}" "${ARGS_EVAL[@]}" "${ARGS_GPU_SPECIFIC[$GPU]}")
		STDOUT="$PROCESS_FOLDER/$FILE_STDOUT"
		STDERR="$PROCESS_FOLDER/$FILE_STDERR"
		echo -e "----GPU_$GPU parameters:"
		echo -e "\tWork folder:\t$PROCESS_FOLDER"
		echo -e "\tstdout log:\t$STDOUT"
		echo -e "\tstderr log:\t$STDERR"
		echo -e "\tCMD (train):\t${CMDLINE_TRAIN[@]}"
		echo -e "\tCMD (eval):\t${CMDLINE_EVAL[@]}"
		echo "#---- Processes for $GPU ----" >> kill_all.sh 
		case $MODE in
			eval|evaluation|decode|decoding)
				echo -e "#### STARTING --==EVALUATION==-- ON GPU_$GPU. . ."
				( echo "kill $BASHPID" >> kill_all.sh; CUDA_VISIBLE_DEVICES=$GPU ${CMDLINE_EVAL[@]} 2>> $STDERR 1>> $STDOUT & echo "kill $!" >> kill_all.sh )
				echo -e "#### PROCESS $! STARTED ON GPU_$GPU"
				;;
			train|training|encode|encoding|*)
				echo -e "#### STARTING --==TRAINING==-- ON GPU_$GPU. . ."
				( echo "kill $BASHPID" >> kill_all.sh; CUDA_VISIBLE_DEVICES=$GPU ${CMDLINE_TRAIN[@]} 2>> $STDERR 1>> $STDOUT & echo "kill $!" >> kill_all.sh )
				echo -e "#### PROCESS $! STARTED ON GPU_$GPU"
				;;
		esac
	fi
done
chmod +x kill_all.sh

