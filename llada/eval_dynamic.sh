export HF_DATASETS_CACHE="/HOME/zyygdlab_shdluo/zyygdlab_shdluo_1/Storage01/zqh/MFTCoder/cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_BASE_URL=https://hf-mirror.com
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE=1

length=512
block_length=32
steps=$((length / block_length))

accelerate launch eval_dynamic.py --tasks gsm8k --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=/HOME/zyygdlab_shdluo/zyygdlab_shdluo_1/Storage01/models/GSAI-ML/LLaDA-8B-Instruct,use_cache=True,dual_cache=True,use_dynamic_block=True,show_speed=True,threshold=0.9,gen_length=${length} \

# accelerate launch eval_dynamic.py --tasks humaneval --num_fewshot 0 \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=/HOME/zyygdlab_shdluo/zyygdlab_shdluo_1/Storage01/models/GSAI-ML/LLaDA-8B-Instruct,use_cache=True,dual_cache=True,use_dynamic_block=True,show_speed=True,threshold=0.9,gen_length=${length} \
# --output_path evals_results/dynamic/humaneval-ns0-${length} --log_samples

# python postprocess_code.py /HOME/zyygdlab_shdluo/zyygdlab_shdluo_1/Storage01/zqh/Fast-dLLM/llada/evals_results/dynamic/humaneval-ns0-512/__HOME__zyygdlab_shdluo__zyygdlab_shdluo_1__Storage01__models__GSAI-ML__LLaDA-8B-Instruct/samples_humaneval_2026-03-18T15-55-18.904501.jsonl

# accelerate launch --main_process_port 29998 eval_dynamic.py --tasks gsm8k --num_fewshot 0 \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='/HOME/zyygdlab_shdluo/zyygdlab_shdluo_1/Storage01/models/GSAI-ML/LLaDA-8B-Instruct',gen_length=1024,block_length=32,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True

# accelerate launch --main_process_port 29999 eval_test.py --tasks gsm8k --num_fewshot 0 \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='/HOME/zyygdlab_shdluo/zyygdlab_shdluo_1/Storage01/models/GSAI-ML/LLaDA-8B-Instruct',block_length=2,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True,predictor_path='/HOME/zyygdlab_shdluo/zyygdlab_shdluo_1/Storage01/zqh/Fast-dLLM/gsm8k_length_predictor.pth'

# accelerate launch eval_llada.py --tasks humaneval \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='/HOME/zyygdlab_shdluo/zyygdlab_shdluo_1/Storage01/models/GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True \
# --output_path evals_results/cache_parallel/humaneval-ns0-${length} --log_samples

