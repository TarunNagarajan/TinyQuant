# TinyPolicy
A fast, quantized neural policy execution runtime in C++ for embedded and real-time control systems.

| Day    | Task                                                      | Technical Goals                                               | Learning Objectives                                 |
| ------ | --------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| **1**  | Initialize GitHub repo and folder structure               | Create `runtime/`, `models/`, `scripts/`, `benchmarks/`, etc. | Learn clean codebase layout for embedded ML systems |
| **2**  | Set up CMake (C++) or Cargo (Rust), basic build test      | Add `main.cpp`, test compilation                              | Build system fluency, binary entrypoint             |
| **3**  | Train a small 2-layer MLP in PyTorch                      | Inputs: angle, velocity; Output: torque                       | Tiny control policy generation                      |
| **4**  | Export MLP as ONNX and raw binary format (weights only)   | `torch.onnx.export()` and/or `.npy/.bin` file                 | ML model export formats                             |
| **5**  | Write Python script to export model config (layers, dims) | Output a `.json` or `.cfg` metadata file                      | Format abstraction between training/runtime         |
| **6**  | Document model formats (ONNX, raw weights, metadata)      | In `README.md` and `models/README.md`                         | Project documentation, reproducibility              |
| **7**  | Write `model_loader.hpp`: loads raw weights + metadata    | Store as aligned `std::vector<float>`                         | Binary parsing, file I/O                            |
| **8**  | Parse ONNX format using flatbuffer/ONNX-CPP or skip       | Optional advanced ONNX loader                                 | ONNX internals (if needed)                          |
| **9**  | Test model loader with dummy input                        | Validate layer sizes, weight matrix shapes                    | Defensive coding, unit sanity                       |
| **10** | Add activation functions (ReLU, tanh)                     | Pure `inline` C++ functions                                   | Functional modeling                                 |
| **11** | Write `inference_engine.hpp`: MLP forward pass            | Matrix-vector multiplication, layer traversal                 | Manual NN inference flow                            |
| **12** | Optimize inference loop: reduce heap allocs               | Use stack preallocation / buffers                             | Memory-efficient loop design                        |
| **13** | Simulate single policy inference step                     | Feed random input, print output                               | Validate correctness                                |
| **14** | Add `main.cpp`: simple CLI that loads and infers          | Input from command line or file                               | Interface design, CLI handling                      |
| **15** | Create `test_inference.cpp` unit test                     | Validate outputs for known weights                            | Unit testing basics                                 |
| **16** | Add real-time scheduler loop: tick every 10ms             | Use `chrono::steady_clock` or `std::thread::sleep`            | Timer accuracy, real-time behavior                  |
| **17** | Simulate control input: `angle`, `velocity` from CSV      | Input reading and loop integration                            | Input preprocessing, CSV parsing                    |
| **18** | Apply NN output to a simple system: θ(t+1) = θ(t) + v\*dt | Simulated control effect                                      | Control dynamics understanding                      |
| **19** | Log outputs to CSV for plotting                           | Add simple CSV logger                                         | Log-based debugging                                 |
| **20** | Plot results: angle over time, torque values              | Use Python/matplotlib                                         | Visualization for control behavior                  |
| **21** | Add noise to inputs (sensor emulation)                    | Random Gaussian noise                                         | Sensor modeling, noise robustness                   |
| **22** | Tune controller behavior by retraining NN                 | Adjust training reward / loss                                 | ML-controller tuning cycle                          |
| **23** | Write `scheduler.hpp`: wraps loop and timing              | Modularize real-time loop                                     | Modular design patterns                             |
| **24** | Refactor loop into: sensor → model → actuator → update    | Clear system architecture                                     | Loop control best practices                         |
| **25** | Add JSON config file for loop rate, model path            | Use `nlohmann/json` or similar                                | Config loading in C++                               |
| **26** | Parse and apply config at runtime                         | Validate CLI + config fallback                                | Robust runtime boot sequence                        |
| **27** | Add logging level controls (info/debug)                   | Implement simple log macro                                    | Logging frameworks                                  |
| **28** | Begin quantized mode: implement fake quantizer            | `float32 → int8` linear transform                             | Quantization intuition                              |
| **29** | Replace float weights with `int8_t`, dequant on load      | Simulate INT8 inference                                       | Memory compression awareness                        |
| **30** | Compare float vs quantized outputs                        | Validate tolerable error                                      | Precision benchmarking                              |
| **31** | Implement fixed-point math for inference                  | Use Q15 format: `int16_t` with scale                          | Manual fixed-point DSP-style ops                    |
| **32** | Add timing macros to profile ops                          | Measure per-layer latency                                     | Microbenchmarking skills                            |
| **33** | Write `latency_test.cpp` to profile full inference        | Compare float vs quantized                                    | Benchmark pipeline flow                             |
| **34** | Use `valgrind` or `perf` to track memory and CPU          | Full-system resource profiling                                | Systems-level performance                           |
| **35** | Optional: flamegraph generation for hot paths             | Use `perf script` and flamegraph.pl                           | CPU time analysis                                   |
| **36** | Add option for warm-up iterations                         | Handle cold-start vs warm-run latency                         | Runtime benchmarking practices                      |
| **37** | Try memory pooling (optional)                             | Avoid allocs in inference path                                | Memory management techniques                        |
| **38** | Simulate varying tick rates (5ms–50ms)                    | Stress test control loop                                      | Timing sensitivity analysis                         |
| **39** | Create CLI switch for real-time vs batch mode             | Support offline benchmarks                                    | Flexible usage modes                                |
| **40** | Add `make benchmark` or `cargo bench` target              | Streamlined measurement workflow                              | Build pipeline ergonomics                           |
| **41** | Inject failure cases (NaN input, file missing)            | Improve error handling                                        | Defensive engineering                               |
| **42** | Write `test_scheduler.cpp`: check tick accuracy           | Timer jitter test                                             | Real-time accuracy validation                       |
| **43** | Create full `tests/` suite                                | Organize, add Makefile target                                 | Testing coverage                                    |
| **44** | Auto-run tests with CI (GitHub Actions optional)          | Test automation                                               | GitHub workflows                                    |
| **45** | Add inline docs to every function                         | Doxygen-compatible comments                                   | API clarity, professional polish                    |
| **46** | Draw system diagram: input → policy → actuator            | SVG or PNG                                                    | Visual documentation                                |
| **47** | Add `README.md`: overview, usage, architecture            | Full documentation draft                                      | Technical writing                                   |
| **48** | Add example output plots (`plots/`)                       | Torque/angle/time series                                      | Show don't tell                                     |
| **49** | Test on low-power mode: simulate slow CPU                 | Limit cores, simulate embedded                                | Performance realism                                 |
| **50** | (Optional) Compile for ARM with cross-compiler            | `arm-linux-gnueabihf-g++`                                     | Embedded toolchain                                  |
| **51** | Refactor for minimal runtime footprint                    | Goal: <100KB binary                                           | Embedded readiness                                  |
| **52** | Package minimal release zip/tar.gz                        | CLI tool ready to ship                                        | Real-world deployment exercise                      |
| **53** | Record screen of CLI + plots                              | Show simulated control loop                                   | Demo creation                                       |
| **54** | Write `examples/` folder: sensor sim + loop               | Extra demos                                                   | Showcase behavior                                   |
| **55** | (Optional) Add WASM target                                | Compile for browser                                           | Web runtime portability                             |
| **56** | (Optional) Write a blog explaining the architecture       | Share online                                                  | Technical storytelling                              |
| **57** | Final testing + bugfixes                                  | All modules reviewed                                          | Ship-ready confidence                               |
| **58** | Polish all docs, add badges, finalize license             | OSS best practices                                            | GitHub portfolio boost                              |
| **59** | Push final commit, write detailed commit message          | Ship milestone v1.0                                           | Versioning                                          |

TinyPolicy/
├── runtime/                # Core C++ inference engine & scheduler
│   ├── main.cpp            # Entry point: loads model, runs loop
│   ├── inference.hpp       # MLP forward pass (quantized, float, etc.)
│   ├── scheduler.hpp       # Real-time tick manager
│   ├── model_loader.hpp    # Parses weights/config from file
│   ├── activations.hpp     # ReLU, tanh, etc.
│   └── utils.hpp           # Logging, timing macros, etc.
│
├── models/                 # Saved model weights and configs
│   ├── tiny_policy.onnx    # Exported from PyTorch
│   ├── weights.bin         # Raw binary weights
│   ├── metadata.json       # Layer sizes, scale, zero-points
│   └── README.md
│
├── scripts/                # Python tools (training, export, plotting)
│   ├── train_policy.py     # Tiny MLP for simple control task
│   ├── export_to_onnx.py   # Converts to ONNX and raw format
│   ├── plot_logs.py        # Reads CSV log, plots outputs
│   └── generate_inputs.py  # Simulates noisy sensor inputs
│
├── benchmarks/             # Benchmark + profiling utilities
│   ├── latency_test.cpp
│   ├── mem_profile.md
│   └── perf_results.csv
│
├── tests/                  # Unit tests for components
│   ├── test_inference.cpp
│   ├── test_scheduler.cpp
│   └── test_loader.cpp
│
├── data/                   # Sample inputs for inference
│   ├── inputs.csv
│   └── outputs_expected.csv
│
├── plots/                  # Output plots (generated from logs)
│   ├── angle_vs_time.png
│   ├── torque_output.png
│   └── README.md
│
├── CMakeLists.txt          # Build config (can add subdirs)
├── README.md               # Project overview, usage, performance
├── LICENSE                 # MIT, Apache 2.0, or your choice
└── .gitignore
