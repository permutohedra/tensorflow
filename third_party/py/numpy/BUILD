licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "numpy",
    srcs = ["tf_numpy_dummy.py"],
    srcs_version = "PY2AND3",
)

# Use a mechanism from the workspace to import Python header files.
cc_library(
    name = "headers",
    deps = ["//external:numpy_headers"],
)

#cc_library(
#    name = "headers",
#    hdrs = glob(["numpy_include/**/*.h"]),
#    data = ["//util/python:python_checked"],
#    includes = ["numpy_include"],
#)

genrule(
    name = "dummy",
    outs = ["tf_numpy_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
