MODEL_ID="facebook/opt-350m"
OUTPUT_FILE="generation_attn.json"
MAX_LENGTH="50"
TOPK="5"

Instruction="Give me a simple explanation of the large language model."

python gen.py \
    --model_id $MODEL_ID \
    --input_text "$Instruction" \
    --output_file $OUTPUT_FILE \
    --max_length $MAX_LENGTH \
    --top_k $TOPK \
