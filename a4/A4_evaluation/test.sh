#!/bin/bash

# Base directory (relative to script location)
BASE_DIR="submissions"
INPUT_DIR="$BASE_DIR/input"

# Resolve full path to input directory
INPUT_DIR_FULL=$(realpath "$INPUT_DIR")

# Timeout settings
TIMEOUT_CMD="timeout"    # requires GNU coreutils timeout
TIMEOUT_DUR="10m"        # 10 minutes per testcase

# Loop over each student folder (excluding 'input')
for STUDENT_DIR in "$BASE_DIR"/*; do
    if [ -d "$STUDENT_DIR" ] && [ "$(basename "$STUDENT_DIR")" != "input" ]; then
        echo "Processing $(basename "$STUDENT_DIR")..."

        cd "$STUDENT_DIR" || continue

        # Compile main.cu
        if nvcc -arch=sm_75 -std=c++17 -rdc=true main.cu -o a.out; then
            echo "Compilation successful."
        else
            echo "Compilation failed for $(basename "$STUDENT_DIR"), skipping."
            cd - >/dev/null
            continue
        fi

        # Create output directory if not exists
        mkdir -p output

        # Loop over input files
        for i in {0..39}; do
            INPUT_FILE="$INPUT_DIR_FULL/input$i.txt"
            OUTPUT_FILE="output/output$i.txt"
            TIMEOUT_FILE="output/timeout_${i}.txt"

            # Skip if already processed
            if [ -f "$OUTPUT_FILE" ] || [ -f "$TIMEOUT_FILE" ]; then
                echo "Skipping testcase $i (already processed)."
                echo "Completed testcase $i."
                continue
            fi

            if [ -f "$INPUT_FILE" ]; then
                # Run with a timeout, suppressing any stdout/stderr
                $TIMEOUT_CMD $TIMEOUT_DUR ./a.out "$INPUT_FILE" "$OUTPUT_FILE" &>/dev/null
                EXIT_CODE=$?

                if [ $EXIT_CODE -ne 0 ]; then
                    # Any error or timeout: remove any output and create blank timeout file
                    rm -f "$OUTPUT_FILE"
                    touch "$TIMEOUT_FILE"
                    echo "Testcase $i failed or timed out (exit code $EXIT_CODE); created blank $(basename "$TIMEOUT_FILE")."
                else
                    # Success: keep the output file
                    echo "Testcase $i ran successfully. Output is in $(basename "$OUTPUT_FILE")."
                fi
            else
                echo "Input file for testcase $i not found; skipping."
            fi

            # Mark completion
            echo "Completed testcase $i."
        done

        cd - >/dev/null
    fi
done

echo "All processing complete."
