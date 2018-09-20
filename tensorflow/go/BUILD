# Description:
# Go API for TensorFlow.

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = 'srcs',
    srcs = ['doc.go'],
    visibility = ['//visibility:public'],
)

filegroup(
    name = 'cgo_srcs',
    srcs = glob(
        ['*.go'],
        exclude = [
            'doc.go',
            '*_test.go',
        ],
    ),
    visibility = ['//visibility:public'],
)

sh_test(
    name = "test",
    size = "small",
    srcs = ["test.sh"],
    data = [
        ":all_files",  # Go sources
        "//tensorflow:libtensorflow.so",  # C library
        "//tensorflow/c:headers",  # C library header
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",  # Testdata for LoadSavedModel
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
