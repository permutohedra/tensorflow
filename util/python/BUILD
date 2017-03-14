licenses(["notice"])  # New BSD, Python Software Foundation

package(default_visibility = ["//visibility:public"])

# Use a mechanism from the workspace to import Python header files.
cc_library(
  name = "python_headers",
  deps = ["//external:python_headers"],
)

#cc_library(
#    name = "python_headers",
#    hdrs = glob([
#        "python_include/**/*.h",
#    ]),
#    data = [":python_checked"],
#    includes = ["python_include"],
#)
#
#genrule(
#    name = "python_check",
#    srcs = [
#        "python_config.sh",
#        "configure_files",
#    ],
#    outs = [
#        "python_checked",
#    ],
#    cmd = "OUTPUTDIR=\"$(@D)/\"; $(location :python_config.sh) --check && touch $$OUTPUTDIR/python_checked",
#    local = 1,
#)
#
#filegroup(
#    name = "configure_files",
#    data = glob([
#        "*",
#    ]),
#)
