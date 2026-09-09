project('dump_reference', 'fortran')

sdftd3_dep = dependency(
  's-dftd3',
  fallback: ['s-dftd3', 'sdftd3_dep'],
  default_options: [
    'default_library=static',
    'api=false',
    'python=false',
  ],
)

executable(
  'dump_reference',
  'dump_reference.f90',
  dependencies: sdftd3_dep,
  install: false,
)
