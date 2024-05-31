from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(name="pix_layer",
      ext_modules=[
          CUDAExtension('pix_layer_cuda',[
              'pix_layer.cpp',
              'pix_layer_cuda.cu'
          ])
      ],
      cmdclass={
          'build_ext': BuildExtension
      })
