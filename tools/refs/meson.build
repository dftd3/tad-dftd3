project('gen_refs_fortran', 'fortran')

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
  'gen_refs_fortran',
  'gen_refs_fortran.f90',
  dependencies: sdftd3_dep,
  install: true,
)
