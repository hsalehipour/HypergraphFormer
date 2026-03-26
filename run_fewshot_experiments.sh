#!/bin/bash
#
# Run few-shot hypergraph generation experiments and optionally score them.
#
# Calls scripts/fewshot_generator.py in batch mode for each (backend, shots,
# round) combination, then optionally runs scripts/score_results.py on
# every output file.
#
# Usage:
#   # GPT experiments (1/2/3-shot, 5 rounds each)
#   bash run_fewshot_experiments.sh --backend gpt --examples-dir eval_format/train
#
#   # Qwen experiments with custom shots and rounds
#   bash run_fewshot_experiments.sh --backend qwen --shots 1 3 5 --rounds 3
#
#   # Generate + score with GED afterwards
#   bash run_fewshot_experiments.sh --backend gpt --score --test-dir dataset/hypergraph/test --metrics ged
#
#   # Score only (skip generation, just score existing results)
#   bash run_fewshot_experiments.sh --score-only --test-dir dataset/hypergraph/test --metrics ged accuracy
#
#   # All options
#   bash run_fewshot_experiments.sh \
#       --backend gpt --model gpt-4-turbo \
#       --examples-dir eval_format/train \
#       --shots 1 2 3 --rounds 5 \
#       --max-tokens 8192 --temperature 0.7 \
#       --output-dir gpt_few_shots \
#       --score --test-dir dataset/hypergraph/test --metrics ged accuracy delta

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
BACKEND=""
MODEL=""
EXAMPLES_DIR="eval_format/train"
SHOTS=(1 2 3)
ROUNDS=5
MAX_TOKENS=8192
TEMPERATURE=0.7
OUTPUT_DIR="fewshot_results"
SCORE=false
SCORE_ONLY=false
TEST_DIR=""
METRICS=""

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)      BACKEND="$2";       shift 2 ;;
        --model)        MODEL="$2";         shift 2 ;;
        --examples-dir) EXAMPLES_DIR="$2";  shift 2 ;;
        --shots)
            SHOTS=()
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SHOTS+=("$1"); shift
            done
            ;;
        --rounds)       ROUNDS="$2";        shift 2 ;;
        --max-tokens)   MAX_TOKENS="$2";    shift 2 ;;
        --temperature)  TEMPERATURE="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";    shift 2 ;;
        --score)        SCORE=true;         shift ;;
        --score-only)   SCORE_ONLY=true; SCORE=true; shift ;;
        --test-dir)     TEST_DIR="$2";      shift 2 ;;
        --metrics)
            METRICS=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                METRICS="$METRICS $1"; shift
            done
            ;;
        -h|--help)
            head -30 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $SCORE && [[ -z "$TEST_DIR" || -z "$METRICS" ]]; then
    echo "Error: --score requires --test-dir and --metrics"
    exit 1
fi

if ! $SCORE_ONLY && [[ -z "$BACKEND" ]]; then
    echo "Error: --backend is required (qwen or gpt)"
    exit 1
fi

TOTAL=$(( ${#SHOTS[@]} * ROUNDS ))

# ── Generation phase ──────────────────────────────────────────────────
if ! $SCORE_ONLY; then
    echo "========================================================================"
    echo "  Few-Shot Experiments: backend=$BACKEND, shots=${SHOTS[*]}, rounds=$ROUNDS"
    echo "  Output: $OUTPUT_DIR/"
    echo "========================================================================"

    RUN=0
    FAILED=0

    for SHOT in "${SHOTS[@]}"; do
        SUFFIX="shot"
        [[ "$SHOT" -gt 1 ]] && SUFFIX="shots"
        SHOT_DIR="$OUTPUT_DIR/${SHOT}_${SUFFIX}"
        mkdir -p "$SHOT_DIR"

        for R in $(seq 1 "$ROUNDS"); do
            RUN=$((RUN + 1))
            OUTPUT_FILE="$SHOT_DIR/hypergraph_round_${R}.json"
            echo -ne "\r[$RUN/$TOTAL] ${SHOT}-shot round $R ..."

            CMD=(python scripts/fewshot_generator.py
                --backend "$BACKEND"
                --examples-dir "$EXAMPLES_DIR"
                --num-examples "$SHOT"
                --max-tokens "$MAX_TOKENS"
                --temperature "$TEMPERATURE"
                --output "$OUTPUT_FILE"
                --from-dataset
            )
            [[ -n "$MODEL" ]] && CMD+=(--model "$MODEL")

            if "${CMD[@]}" > /dev/null 2>&1; then
                echo -ne "\r[$RUN/$TOTAL] ${SHOT}-shot round $R ... OK          "
            else
                echo -ne "\r[$RUN/$TOTAL] ${SHOT}-shot round $R ... FAILED      "
                FAILED=$((FAILED + 1))
            fi
            echo ""
        done
    done

    echo ""
    echo "========================================================================"
    echo "  Done: $((TOTAL - FAILED))/$TOTAL succeeded"
    echo "========================================================================"
fi

# ── Scoring phase ─────────────────────────────────────────────────────
if $SCORE; then
    echo ""
    echo "========================================================================"
    echo "  Scoring results in $OUTPUT_DIR/ with metrics:$METRICS"
    echo "========================================================================"

    for RESULTS_FILE in "$OUTPUT_DIR"/*/*.json; do
        [[ -f "$RESULTS_FILE" ]] || continue
        echo ""
        echo "--- $RESULTS_FILE ---"
        python scripts/score_results.py \
            --results-file "$RESULTS_FILE" \
            --test-dir "$TEST_DIR" \
            --metrics $METRICS \
            --summary-only \
        || echo "  (scoring failed)"
    done

    echo ""
    echo "========================================================================"
    echo "  Scoring complete"
    echo "========================================================================"
fi
