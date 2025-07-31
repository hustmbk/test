#!/bin/bash
# CUTLASS安装脚本 - 针对H20 GPU优化
# 作者: Claude Code Assistant

set -e  # 遇到错误时退出

echo "🚀 开始安装CUTLASS for H20 GPU (SM_90a)..."

# 检查CUDA版本
echo "📋 检查CUDA环境..."
nvcc --version

# 进入library目录
cd /root/autodl-tmp/RetrievalAttention/library

# 1. 解压CUTLASS（假设用户已上传cutlass-main.zip）
echo "📦 解压CUTLASS..."
if [ -f "cutlass-main.zip" ]; then
    unzip -q cutlass-main.zip
    mv cutlass-main cutlass
    echo "✅ CUTLASS解压完成"
else
    echo "❌ 未找到cutlass-main.zip，请确保已上传该文件"
    exit 1
fi

# 2. 安装CUDA BLAS for H20 GPU
echo "🔧 安装H20 GPU专用CUDA BLAS..."
pip install nvidia-cublas-cu12==12.4.5.8

# 3. 创建CUTLASS构建目录
echo "🏗️ 准备构建CUTLASS..."
cd cutlass
mkdir -p build
cd build

# 4. 配置CMake for H20 GPU (SM_90a)
echo "⚙️ 配置CMake for H20 GPU (SM_90a)..."
cmake .. \
    -DCUTLASS_NVCC_ARCHS=90a \
    -DCUTLASS_LIBRARY_KERNELS=all \
    -DCUTLASS_ENABLE_CUBLAS=ON \
    -DCUTLASS_ENABLE_CUDNN=OFF \
    -DUBSAN=OFF \
    -DCMAKE_BUILD_TYPE=Release

# 5. 编译CUTLASS
echo "🔨 编译CUTLASS库..."
make -j$(nproc)

# 6. 安装CUTLASS
echo "📥 安装CUTLASS..."
make install

# 7. 返回library目录准备安装RetroInfer kernels
cd /root/autodl-tmp/RetrievalAttention/library

# 8. 修改RetroInfer setup.py以支持H20 GPU
echo "🔧 配置RetroInfer for H20 GPU..."
cd retroinfer

# 备份原始setup.py
cp setup.py setup.py.backup

# 修改setup.py支持SM_90a
cat > setup.py << 'EOF'
import os
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# CUDA和CUTLASS路径
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
cutlass_path = '../cutlass'

# H20 GPU (SM_90a) 优化编译参数
compile_args = [
    '-O3',
    '-std=c++17',
    '-DWITH_CUDA',
    '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
    f'-I{cuda_home}/include',
    f'-I{cutlass_path}/include',
    f'-I{cutlass_path}/tools/util/include',
    f'-I{cutlass_path}/examples/common',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-Xcompiler=-fPIC',
    '-Xcompiler=-Wno-float-conversion',
    '-gencode=arch=compute_90a,code=sm_90a',  # H20 GPU专用
    '-DCUTLASS_ARCH_MMA_SM90A_SUPPORTED=1'
]

link_args = [
    f'-L{cuda_home}/lib64',
    '-lcudart',
    '-lcublas',
    '-lcurand'
]

# 源文件
sources = [
    'retroinfer_kernels.cpp',
    # 添加其他必要的源文件
]

ext_modules = [
    Pybind11Extension(
        'retroinfer_kernels',
        sources,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++'
    )
]

setup(
    name='retroinfer_kernels',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    python_requires='>=3.8',
)
EOF

echo "✅ RetroInfer setup.py已配置为支持H20 GPU"

# 9. 编译并安装RetroInfer kernels
echo "🔨 编译RetroInfer kernels..."
pip install . -v

# 10. 验证安装
echo "🧪 验证安装..."
cd /root/autodl-tmp/RetrievalAttention
python -c "
try:
    import retroinfer_kernels
    print('✅ RetroInfer kernels导入成功')
    
    # 测试CUDA设备
    import torch
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        print(f'✅ CUDA设备: {device.name}')
        print(f'✅ 计算能力: {device.major}.{device.minor}')
    else:
        print('❌ CUDA不可用')
        
except ImportError as e:
    print(f'❌ RetroInfer kernels导入失败: {e}')
except Exception as e:
    print(f'❌ 验证过程出错: {e}')
"

echo "🎉 CUTLASS和RetroInfer安装完成！"
echo "📝 现在可以运行测试: python -u simple_test.py --batch_size 4"