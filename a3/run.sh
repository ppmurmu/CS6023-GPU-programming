current_dir=$(pwd)
INPUT="$current_dir/input"
OUTPUT="$current_dir/output"
SUBMIT="$current_dir/submit"

touch logFile
touch timing_logFile

echo "======= START ========="
echo "======= START =========" >> $current_dir/logFile
date >> $current_dir/logFile

cd $SUBMIT
ROLLNO=$(ls *.cu | tail -1 | cut -d'.' -f1)
echo "$ROLLNO"
cp ${ROLLNO}.cu main.cu

# Compile the CUDA file
bash compile.sh

# Check if compilation was successful
if [ ! -f main.out ]; then
    echo "Compilation failed, main.out not generated."
    echo "Compilation failed, main.out not generated." >> $current_dir/logFile
    echo "========== END ========="
    echo "========== END =========" >> $current_dir/logFile
    exit 1
fi

# Initialize counters for summary
success_count=0
failure_count=0

# Iterate over each test case
for testcase in $INPUT/*; do
    filename=${testcase##*/}

    # Run the main program and capture output
    ./main.out < $INPUT/$filename >> $current_dir/logFile

    # Compare the output with the expected output
    diff $OUTPUT/$filename $SUBMIT/cuda.out -b > /dev/null 2>&1
    exit_code=$?

    # Log success or failure
    if ((exit_code == 0)); then
        echo "$filename success"
        echo "$filename success" >> $current_dir/logFile
        ((success_count++))
    else
        echo "$filename failure"
        echo "$filename failure" >> $current_dir/logFile
        ((failure_count++))
    fi

    # Log timing output in a simplified format (filename: time)
    if [ -f $SUBMIT/cuda_timing.out ]; then
        timing=$(cat $SUBMIT/cuda_timing.out)
        echo "$filename: $timing" >> $current_dir/timing_logFile
    else
        echo "$filename: No timing output found" >> $current_dir/timing_logFile
    fi
done

# Summary of results
echo "Summary: $success_count success, $failure_count failure" >> $current_dir/logFile

echo "========== END ========="
echo "========== END =========" >> $current_dir/logFile
