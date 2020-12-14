This project is archived. Reach out to authors by email for questions.

# Build

Both Linux and Windows 10 builds use Intel(R) MKL(R) library. We used 2017 version; other versions might work as well.
Suppose you have installed MKL at `<MKL_ROOT>`. In Linux, this is typically `/opt/intel/mkl`.
In Windows 10, this is typically `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017\windows\mkl`

## Linux / gcc
We built this project on Ubuntu 16.04LTS with gcc 5.4. Other linux versions with gcc 5+ could also work.
```
cd <ISLE_ROOT>
export LD_LIBRARY_PATH=<MKL_ROOT>/lib/intel64/:.
make -j
```
This should generate two executables `ISLETrain` and `ISLEInfer` in the `<ISLE_ROOT>` directory.


## Windows 10 / Visual Studio 2015

Open `<ISLE_ROOT>\win\ISLE.sln` in VS2015, and build the `ISLETrain` and `ISLEInfer` projects.
The Debug build uses dynamic linking for C runtime, OpenMP, and MKL libraries.
The Release build links all these dependencies statically into a self-contained exe.

If there is a problem with this, you could configure project file properties as follows:
* Under VC++ Directories >
  * To Include directories, add `<ISLE_ROOT>` and `<ISLE_ROOT>\include`. This will include `include`,  `Eigen` and `Spectra` directories.
  * To Library Directories, add `<MKL_ROOT>\lib\intel64_win` (for MKL libs) and `<MKL_ROOT>\..\compiler\lib\intel64_win\` (for OpenMP lib)

* Under C/C++ >
  * Enable optimizations, and disable runtime and SCL checks (this conflicts with turning on optimizations).
  * Code Generation > Enable Parallel Code
  * Languages > Enable OpenMP support

* Under 'Intel Performance Libraries' >
  * Enable 'Parallel' option.
  * Enable MKL_ILP64 option for 8-byte MKL_INTs.

* Under Linker > Input > Additional dependencies: Add
  * Add the prefix `libiomp5md.lib;mkl_core.lib;mkl_intel_ilp64.lib;mkl_sequential.lib;` to existing libraries

# Training on a dataset

1. Create a `<tdf_file>` which has one `<doc_id> <word_id> <frequency>` entry per line.
   * The `<doc_id>` entries are 1-based and range between 1 and `<num_docs>`. 
   * The `<word_id>` enties are 1-based and range between 1 and `<vocab_size>`.
   * Let `<max_entries>` be the number of entries (or lines) in this file.

2. Create a `<vocab_file>` file with mapping from word id to the word; the i-th line of the file is the word with word_id=i.

3. Run 
 ```
 ISLETrain <tdf_file> <vocab_file> <output_dir> <vocab_size> <num_docs> <max_entries> <num_topics> <sample(0/1)> <sample_rate> <edge topics(0/1) <max_edge_topics>
 ```
   * `<num_topics>` is the number of topics you want to recover from the `<tdf_file>`.
   * If the dataset is too large and you wish to use importance sampling, set `<sample>` to 1 (otherwise 0).
   * When `<sample>` is enabled by setting it to 1, you can specify the sampling rate with `<sample_rate>`. For example, 0.1.
   * when `<edge topic>` is set to one, a larger "edge" topic model with up to `<max_edge_topics>` will be computed.
   * The output will be stored in a log directory under `<output_dir>`

# Inference for a dataset using the trained model

1. Locate the trained `<sparse_model_file>` in the log directory under the trainer's `<output_dir>`.

2. Prepare `<infer_file>` for the documents you want to infer in the same way `<tdf_file>` was prepared from training. `<vocab_size>`, `<num_docs_in_infer_file>` and  `<nnzs_in_infer_file>` are the vocabulary size, the number of documents in the `<infer_file>`, and
 the number of entries(lines) in the `<infer_file>` respectively. Note that the `<vocab_size>` of the infer file can not be larger than the `<vocab_size>` of the `<tdf_file>` on which the model was trained.

3. Run
```
ISLEInfer <sparse_model_file> <infer_file> <output_dir> <num_topics> <vocab_size> <num_docs_in_infer_file> <nnzs_in_infer_file> <nnzs_in_sparse_model_file> <iters> <Lifschitz_constant>
```
   * `<output_dir>` is the location where the file containing the inferred weights is written.
   * `<num_topics>` is the number of topics in the trained model.
   * `<nnzs_in_sparse_model_file>` is the number of entries (lines) in `<sparse_model_file>`.
   * `<iters>` is the number if iterations of multiplicative weight update. Set to 0 to use default number of iterations.
   * `<Lifschitz_constant>` is an estimate of the Lifschitz constant for the gradient of the log-likelihood terms. Set to 0 to use default.


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
