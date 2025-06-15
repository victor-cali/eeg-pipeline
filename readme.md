# Parallel Programming Final Project

## Real-Time Multichannel EEG Preprocessing Pipeline with LabStreamingLayer

This project demonstrates a C++ application receiving data from a Lab Streaming Layer (LSL) stream. The following instructions will guide you through compiling and running both the LSL sender example and the receiver application.

### Prerequisites

Before you begin, ensure you have the following installed:

- Git for cloning the repository.
- A modern CMake (version 3.16 or newer is recommended).
- A C++17 compliant compiler (e.g., a recent version of Clang on macOS, GCC on Linux, or MSVC on Windows).

#### 1. Clone the Repository

First, clone the repository. This project uses liblsl as a git submodule, so you must use the --recursive flag to clone it at the same time.

```bash
git clone --recursive https://github.com/victor-cali/eeg-pipeline
cd eeg-pipeline
```

If you have already cloned the repository without the submodule, you can initialize it by running: git submodule update --init --recursive

#### 2. Build the LSL Sender (SendDataSimple) and the LSL Receiver (ReceiveDataSimple)

You need both a Sender and Receiver to simulate an actual EEG setup. We will build the two of them from the examples included with the liblsl library. This only needs to be done once.

You will need three separate terminal windows for the final step.

In the first terminal:
Navigate to the liblsl directory and create a build folder

```bash
cd external/liblsl
mkdir build && cd build
```

Configure the build with CMake. Crucially, you must enable the examples with:

```bash
cmake -DLSL_BUILD_EXAMPLES=ON ..
```

Compile the library and examples:

```bash
make
```

This will build the liblsl library and all its examples.

#### 3. Build the LSLDemo Application

Now, in the second terminal, we will build our main application.

Configure and build your application with CMake (this must be done in the project's root)

```bash
mkdir build && cd build
```

For Mac:

```bash
cmake -D CMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
      -D CMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
      ..
make
```

This creates the lsl_demo executable inside the build directory.

#### 4. Running the Demonstration

This is the final step where you will see everything working together.
In your first terminal, run the LSL sender program. This will start a broadcast on your local network.

From within the eeg-pipeline/external/liblsl/build/ directory:

```bash
./examples/SendDataMinimal
```

In the second terminal, you will run the main application that performs the filtering and feature extraction

From the project's root

```bash
cd build
./lsl_demo
```

In the third terminal, you will run the receiver, simulating an actual EEG setup.

From within the eeg-pipeline/external/liblsl/build/ directory:

```bash
./examples/ReceiveDataSimple name ProcessedEEGFeatures
```

You must run the programs in the mentioned order. You can validate that the program is working by looking at the output of each terminal.

In the first one you should see something like this:

```bash
2025-06-14 22:48:09.446 (   0.001s) [           91D12]      netinterfaces.cpp:91    INFO| netif 'lo0' (status: 1, multicast: 32768, broadcast: 0)
2025-06-14 22:48:09.448 (   0.001s) [           91D12]      netinterfaces.cpp:91    INFO| netif 'lo0' (status: 1, multicast: 32768, broadcast: 0)
2025-06-14 22:48:09.448 (   0.001s) [           91D12]      netinterfaces.cpp:102   INFO|       IPv4 addr: 7f000001
```

In the second one something like this:

```bash
Attempting to connect to LSL stream 'SimpleStream'...
Resolving stream with name: SimpleStream...
2025-06-14 22:48:12.514 (   0.002s) [           91D2C]      netinterfaces.cpp:91    INFO| netif 'lo0' (status: 1, multicast: 32768, broadcast: 0)
2025-06-14 22:48:12.515 (   0.002s) [           91D2C]      netinterfaces.cpp:91    INFO| netif 'lo0' (status: 1, multicast: 32768, broadcast: 0)
...
...
...
LSL output stream 'ProcessedEEGFeatures' created and broadcasting.

--- Starting Live Parallel Feature Extraction ---
--- Received Chunk of 102 samples, processing 64 in parallel ---
  Features computed and sent to LSL output stream.
```

And in the third one something like this:

```bash
Now pulling samples...
0.893945 0.786649 5.8632 0 0 0.840758 0.702718 5.83195 0 0 0.858253 0.729126 5.8125 0 0
```

You now have a fully functional LSL data pipeline running!
