# mnist-nn

An impatient implementation of a back-propagating neural network for the MNIST (http://yann.lecun.com/exdb/mnist/) dataset.

When trained with the default configuration (see main.cpp), it achieves ~97% accuracy with the test dataset.

Don't take everything in the source in any way as best practices. My impatience is quite evident in some parts.

# prereq
For the MNIST data, either download and decompress the files yourself and place them in a `mnist/` directory in the project root, or do `make mnist-dl`, which will do that for you. This make rule does NOT run as part of the build process.

# usage
To use, build with `make`, `make release`, or `make debug`, and then:
```shell
$ ./build/neural_network train -m <file name>
$ ./build/neural_network predict -m <file name>
```

Or use Visual Studio to build (project outputs the executable to `build/<arch>/<config>/neural_network.exe`).

Requires a C++ 17 capable compiler (and a standard library with `<filesystem>`).

# License
MIT
