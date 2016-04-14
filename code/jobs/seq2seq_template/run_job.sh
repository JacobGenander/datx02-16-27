
PATH_CWD="."
PATH_PYTHON=$(which python2)
PATH_SCRIPT="../../seq2seq/translate.py"
FILE_STDOUT="stdout"
FILE_STDERR="stderr"

# Grab mode as first argument, can be either 'train' or 'eval' 
MODE="$1"

# Arguments that are common for all GPU:s
ARGS_COMMON=( \
	--data_dir ~/ml_data/ \
	--article_file articles_100000.txt \
	--title_file titles_100000.txt \
	--article_vocab_size 2500 \
	--title_vocab_size 2000 \
	--num_layers 2 \
	--train_dir ~/ml_data/results/ \
	--steps_per_checkpoint 10
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
ARGS_GPU_SPECIFIC[0]="--size 512"
ARGS_GPU_SPECIFIC[1]=""
ARGS_GPU_SPECIFIC[2]=""
ARGS_GPU_SPECIFIC[3]="--size 256"

# Used to interate
let HIGHEST_INDEX=${#ARGS_GPU_SPECIFIC[@]}-1

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
		CMDLINE_TRAIN=("$PATH_PYTHON" "$PATH_SCRIPT" "${ARGS_COMMON[@]}" "${ARGS_TRAIN[@]}" "${ARGS_GPU_SPECIFIC[$GPU]}")
		CMDLINE_EVAL=("$PATH_PYTHON" "$PATH_SCRIPT" "${ARGS_COMMON[@]}" "${ARGS_EVAL[@]}" "${ARGS_GPU_SPECIFIC[$GPU]}")
		STDOUT="$PROCESS_FOLDER/$FILE_STDOUT"
		STDERR="$PROCESS_FOLDER/$FILE_STDERR"
		echo -e "----GPU_$GPU parameters:"
		echo -e "\tWork folder:\t$PROCESS_FOLDER"
		echo -e "\tstdout log:\t$STDOUT"
		echo -e "\tstderr log:\t$STDERR"
		echo -e "\tCMD (train):\t${CMDLINE_TRAIN[@]}"
		echo -e "\tCMD (eval):\t${CMDLINE_EVAL[@]}"
		case $MODE in
			eval|evaluation|decode|decoding)
				echo -e "#### STARTING --==EVALUATION==-- ON GPU_$GPU. . ."
				( ${CMDLINE_EVAL[@]} 2>> $STDERR 1>> $STDOUT & )
				;;
			train|training|encode|encoding|*)
				echo -e "#### STARTING --==TRAINING==-- ON GPU_$GPU. . ."
				( ${CMDLINE_TRAIN[@]} 2>> $STDERR 1>> $STDOUT & )
				;;
		esac
		echo -e "#### PROCESS STARTED ON GPU_$GPU"
	fi
done

