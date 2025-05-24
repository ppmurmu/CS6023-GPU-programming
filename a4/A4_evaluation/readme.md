# GPU Programming Assignment - Submission & Evaluation Guide

Follow the steps below to correctly submit and evaluate your code.

## 📁 1. Create Submission Folder
Inside the `submission` directory, create a folder named with **your roll number**.

```bash
mkdir submission/<your_roll_number>
```

## 📄 2. Add Your Code
Place your CUDA source file inside the folder you just created. The file **must** be named `main.cu`.

```
submission/
└── <your_roll_number>/
    └── main.cu
```

## 🧪 3. Run Test Script
Navigate to the root directory and execute the `test.sh` script to run your code against all test cases:

```bash
./test.sh
```

This will generate output files for each test case.

## ✅ 4. Run Evaluation Script
Evaluate your output against the expected simulation:

```bash
./evaluate.sh
```

## 📊 5. View Your Score
After evaluation, your score will be saved in the `scores` directory in a file named after your roll number:

```
scores/
└── <your_roll_number>
```

If you encounter any issues during testing or evaluation, please contact the course TAs.
