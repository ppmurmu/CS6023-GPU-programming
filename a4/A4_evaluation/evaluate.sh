#!/bin/bash

# --- Configuration ---
BASE_DIR="submissions"
INPUT_DIR="$BASE_DIR/input"
EVAL_SRC="$BASE_DIR/evaluation.cu"
EVAL_EXEC="$BASE_DIR/a.out"

# Timeout settings for evaluation
TIMEOUT_CMD="timeout"    # requires GNU coreutils timeout
TIMEOUT_DUR="10m"        # 10 minutes per testcase

# Directory to store all score files
SCORES_DIR="$BASE_DIR/scores"

# --- Prepare scores directory ---
echo "Creating scores directory ($SCORES_DIR)..."
mkdir -p "$SCORES_DIR"

# --- 1. Compile the evaluator ---
echo "Compiling evaluator ($EVAL_SRC)..."
if nvcc -arch=sm_75 -std=c++17 -rdc=true "$EVAL_SRC" -o "$EVAL_EXEC"; then
    echo "Evaluator compiled successfully."
else
    echo "ERROR: Failed to compile $EVAL_SRC." >&2
    exit 1
fi

# --- 2. Run evaluation for each student ---
for STUDENT_DIR in "$BASE_DIR"/*; do
    # skip non-directories and the input & scores folders
    [[ ! -d "$STUDENT_DIR" || "$(basename "$STUDENT_DIR")" =~ ^(input|scores)$ ]] && continue

    STUDENT_NAME="$(basename "$STUDENT_DIR")"
    OUTPUT_DIR="$STUDENT_DIR/output"
    SCORE_FILE="$SCORES_DIR/${STUDENT_NAME}.txt"

    # NEW: if score file already exists, skip this student
    if [ -f "$SCORE_FILE" ]; then
        echo "Skipping $STUDENT_NAME: score file already exists."
        continue
    fi

    # skip any student without an output folder
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Skipping $STUDENT_NAME: no output/ folder found."
        continue
    fi

    echo "Evaluating outputs for $STUDENT_NAME..."
    # create or truncate the student's score file
    : > "$SCORE_FILE"

    # for each of the 40 testcases
    for i in {0..39}; do
        IN_F="$INPUT_DIR/input$i.txt"
        OUT_F="$OUTPUT_DIR/output$i.txt"

        if [ -f "$OUT_F" ]; then
            echo "[$STUDENT_NAME] testcase $i: evaluating (up to $TIMEOUT_DUR)..."
            RESULT=""
            EXIT_CODE=0

            # capture output and exit code
            if OUTPUT=$($TIMEOUT_CMD $TIMEOUT_DUR "$EVAL_EXEC" "$IN_F" "$OUT_F" 2>/dev/null); then
                EXIT_CODE=$?
                RESULT="$OUTPUT"
            else
                EXIT_CODE=$?
            fi

            case $EXIT_CODE in
                0)
                    # successful run, record returned numbers
                    echo "testcase $i: $RESULT" >> "$SCORE_FILE"
                    ;;
                124)
                    # timeout
                    echo "testcase $i: 0 0" >> "$SCORE_FILE"
                    ;;
                *)
                    # any other error
                    echo "testcase $i: ERROR" >> "$SCORE_FILE"
                    ;;
            esac
        else
            echo "testcase $i: MISSING_OUTPUT" >> "$SCORE_FILE"
        fi

        echo "[$STUDENT_NAME] testcase $i evaluated."
    done

done

echo "All evaluations complete. Scores are in $SCORES_DIR."
