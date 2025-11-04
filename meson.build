project('text_det', 'cpp',
    version : '1.0',
    default_options : [
        'cpp_std=c++17',
        'warning_level=2',
        'buildtype=release',
        'optimization=3',
        'b_ndebug=if-release',
        'b_lto=true',          
    ]
)

cpp = meson.get_compiler('cpp')
host = host_machine
sys  = host.system()
cpu  = host.cpu_family()

is_darwin = (sys == 'darwin')
is_linux  = (sys == 'linux')
is_clang  = (cpp.get_id() == 'clang' or cpp.get_id() == 'apple-clang')
is_gcc    = (cpp.get_id() == 'gcc')

# ---- Options ----
ort_root     = get_option('onnxruntime_dir')
cpu_opt      = get_option('cpu_opt')
fast_math   = get_option('fast_math')
thinlto     = get_option('thinlto')
use_openmp  = get_option('use_openmp')
rpath_extra  = get_option('install_rpath_extra')
strict_warn = get_option('strict_warnings')

# ---- Compile/Link flags (guarded) ----
cxx_args = []
ld_args  = []

# Smaller/faster binaries
foreach f : ['-fvisibility=hidden', '-ffunction-sections', '-fdata-sections']
    if cpp.has_argument(f)
        cxx_args += f
    endif
endforeach

# Fast-math (safe for inference)
if fast_math
    foreach f : ['-ffast-math', '-fno-math-errno', '-fno-trapping-math', '-ffp-contract=fast']
        if cpp.has_argument(f)
            cxx_args += f
        endif
    endforeach
endif

# ThinLTO on Clang (in addition to b_lto=true)
if thinlto and is_clang
    if cpp.has_argument('-flto=thin')
        cxx_args += '-flto=thin'
    endif
    if cpp.has_link_argument('-flto=thin')
        ld_args += '-flto=thin'
    endif
endif

# CPU tuning
if cpu_opt != 'generic'
    if cpu == 'x86_64'
        optflag = (cpu_opt == 'native') ? '-march=native' : '-march=' + cpu_opt
        if cpp.has_argument(optflag)
            cxx_args += optflag
        endif
        if cpp.has_argument('-mtune=native')
            cxx_args += '-mtune=native'
        endif
    elif cpu == 'aarch64'
        optflag = (cpu_opt == 'native') ? '-mcpu=native' : '-mcpu=' + cpu_opt
        if cpp.has_argument(optflag)
            cxx_args += optflag
        endif
        if cpp.has_argument('-mtune=native')
            cxx_args += '-mtune=native'
        endif
    endif
endif

# Linker dead-stripping
if is_linux
    if cpp.has_link_argument('-Wl,--gc-sections')
        ld_args += '-Wl,--gc-sections'
    endif
elif is_darwin
    foreach f : ['-Wl,-dead_strip', '-Wl,-dead_strip_dylibs', '-Wl,-headerpad_max_install_names']
        if cpp.has_link_argument(f)
            ld_args += f
        endif
    endforeach
endif

# Optional extra warnings
if strict_warn
    foreach f : ['-Wall', '-Wextra', '-Wpedantic', '-Wconversion', '-Wshadow']
        if cpp.has_argument(f)
            cxx_args += f
        endif
    endforeach
endif

# ---- Dependencies ----
opencv_dep = dependency('opencv4', required : true)

# ONNX Runtime: prefer CMake package; else use -Donnxruntime_dir
ort_dep = dependency('onnxruntime', method : 'cmake', required : false)
ort_inc = []
ort_lib_dir_for_rpath = ''

if not ort_dep.found()
    if ort_root == ''
        error('ONNX Runtime not found via CMake and onnxruntime_dir is empty. ' +
            'Install ORT (e.g., Homebrew on macOS) or pass -Donnxruntime_dir=/path')
    endif
    ort_inc = [include_directories(join_paths(ort_root, 'include'))]
    ort_lib_dir_for_rpath = join_paths(ort_root, 'lib')
    ort_dep = cpp.find_library('onnxruntime', dirs : [ort_lib_dir_for_rpath], required : true)
else
    # If CMake package is found, we still set common Homebrew rpaths on macOS below
    # Exact lib dir from the CMake dep may be non-trivial to query; brew path works well
endif

# OpenMP (tile-level parallelism)
openmp_dep = []
if use_openmp
    omp_try = dependency('openmp', required : false)
    if not omp_try.found()
        error('OpenMP requested but not found. On macOS: brew install libomp. ' +
            'Or disable with -Duse_openmp=false')
    endif
    openmp_dep = [omp_try]
endif

# ---- RPATHs (build + install) ----
rpaths = []

if is_darwin
    # Typical Homebrew locations for libomp and onnxruntime
    rpaths += ['/opt/homebrew/opt/libomp/lib', '/opt/homebrew/opt/onnxruntime/lib']
endif

if ort_lib_dir_for_rpath != ''
    rpaths += [ort_lib_dir_for_rpath]
endif

if rpath_extra != ''
    rpaths += rpath_extra.split(':')
endif

build_rpath_str   = ''
install_rpath_str = ''

if rpaths.length() > 0
    build_rpath_str   = ':'.join(rpaths)
    install_rpath_str = ':'.join(rpaths)
endif

# ---- Sources ----
srcs = [
    'src/main.cpp',
    'src/cli.cpp',
    'src/dbnet.cpp',
    'src/geometry.cpp',
    'src/omp_config.cpp',
    'src/tiling.cpp',
    'src/nms.cpp',
    'src/drawing.cpp',
    'src/run_single.cpp',
    'src/run_bench.cpp'
]

# ---- Target ----
deps = [
    opencv_dep, 
    ort_dep, 
    openmp_dep
] 

inc_dirs = [
    include_directories('include'), 
    ort_inc
]

executable('text_det',
  srcs,
  include_directories : inc_dirs,
  cpp_args  : cxx_args,
  link_args : ld_args,
  dependencies : deps,
  build_rpath : build_rpath_str,  
  install_rpath : install_rpath_str,
  install : true
)

# ---- Summary ----
summary(
    {
        'System' : sys, 
        'CPU' : cpu, 
        'Compiler' : cpp.get_id()
    }, 
    section : 'Env', 
    bool_yn : true
)

summary(
    {
        'OpenCV' : opencv_dep.found(), 
        'ONNXRuntime' : ort_dep.found(), 
        'OpenMP' : use_openmp
    }, 
    section : 'Deps', 
    bool_yn : true
)

summary(
    {
        'cpu_opt' : cpu_opt, 
        'fast_math' : fast_math, 
        'thinlto' : thinlto, 
        'lto' : get_option('b_lto')
    }, 
    section : 'Build'

)

if install_rpath_str != ''
    summary(
        {
            'install_rpath' : install_rpath_str
        }, 
        section : 'RPATH'
    )
endif
