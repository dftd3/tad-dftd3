! This file is part of tad-dftd3.
! SPDX-Identifier: Apache-2.0
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

!> Small standalone tool that dumps the coordination number, reference
!> weights and atomic C6 coefficients for one molecular structure, computed
!> by the s-dftd3 Fortran library itself -- exactly the three quantities its
!> Python bindings do not expose (see Task A7 in IMPLEMENTATION_PLAN.md).
!>
!> This program calls the same three library routines, in the same order,
!> that s-dftd3's own command-line tool uses in `property_calc` (in
!> `app/driver.f90`, around the `get_coordination_number` /
!> `weight_references` / `get_atomic_c6` calls) -- the tool exists only to
!> expose their result as JSON, not to recompute anything differently.
!>
!> This program is molecular only (no periodic boundary conditions): the
!> lattice-point list handed to `get_coordination_number` is a single point
!> at the origin, which is the correct input for a finite system.
!>
!> Input is read from standard input, as plain whitespace-separated text:
!>
!>     <number of atoms>
!>     <coordination-number cutoff, in Bohr>
!>     <atomic number> <x> <y> <z>   (repeated once per atom, x/y/z in Bohr)
!>
!> Output is a single line of JSON on standard output:
!>
!>     {"cn": [...], "weights": [[...], ...], "c6": [[...], ...]}
!>
!> `weights` has shape (nat, mref) and `c6` has shape (nat, nat), both
!> written out row-major (one JSON array per atom), which is the shape and
!> order `numpy.array(...)` reconstructs directly with no reshaping needed
!> on the Python side.
program dump_reference
   use, intrinsic :: iso_fortran_env, only : output_unit, error_unit, input_unit
   use mctc_env, only : wp
   use mctc_io, only : structure_type, new
   use dftd3, only : d3_model, new_d3_model, get_coordination_number
   implicit none

   type(structure_type) :: mol
   type(d3_model) :: disp

   integer :: nat, iat, mref
   integer, allocatable :: num(:)
   real(wp), allocatable :: xyz(:, :)
   real(wp) :: cutoff

   real(wp), allocatable :: cn(:), gwvec(:, :), c6(:, :)
   real(wp) :: origin(3, 1)

   read(input_unit, *) nat
   read(input_unit, *) cutoff

   allocate(num(nat), xyz(3, nat))
   do iat = 1, nat
      read(input_unit, *) num(iat), xyz(1, iat), xyz(2, iat), xyz(3, iat)
   end do

   call new(mol, num, xyz)
   call new_d3_model(disp, mol)

   mref = maxval(disp%ref)
   allocate(cn(nat), gwvec(mref, nat), c6(nat, nat))

   ! A single lattice point at the origin: the correct (and only) "lattice
   ! translation" a finite, non-periodic structure needs.
   origin = 0.0_wp

   call get_coordination_number(mol, origin, cutoff, disp%rcov, cn)
   call disp%weight_references(mol, cn, gwvec)
   call disp%get_atomic_c6(mol, gwvec, c6=c6)

   call write_json(output_unit, cn, gwvec, c6)

contains

   !> Write the three arrays as one line of JSON, at full double precision.
   subroutine write_json(unit, cn, gwvec, c6)
      integer, intent(in) :: unit
      real(wp), intent(in) :: cn(:)
      real(wp), intent(in) :: gwvec(:, :)
      real(wp), intent(in) :: c6(:, :)

      integer :: i, n, m

      n = size(cn)
      m = size(gwvec, 1)

      write(unit, "(a)", advance="no") '{"cn": '
      call write_vector(unit, cn)

      write(unit, "(a)", advance="no") ', "weights": ['
      do i = 1, n
         if (i > 1) write(unit, "(a)", advance="no") ", "
         ! gwvec is (mref, nat); write atom i's column as one JSON row.
         call write_vector(unit, gwvec(:, i))
      end do
      write(unit, "(a)", advance="no") "]"

      write(unit, "(a)", advance="no") ', "c6": ['
      do i = 1, n
         if (i > 1) write(unit, "(a)", advance="no") ", "
         call write_vector(unit, c6(:, i))
      end do
      write(unit, "(a)", advance="no") "]"

      write(unit, "(a)") "}"
   end subroutine write_json

   !> Write one real(wp) vector as a JSON array, at full double precision.
   subroutine write_vector(unit, vec)
      integer, intent(in) :: unit
      real(wp), intent(in) :: vec(:)

      integer :: k
      character(len=32) :: buffer

      write(unit, "(a)", advance="no") "["
      do k = 1, size(vec)
         if (k > 1) write(unit, "(a)", advance="no") ", "
         write(buffer, "(es24.16e3)") vec(k)
         write(unit, "(a)", advance="no") trim(adjustl(buffer))
      end do
      write(unit, "(a)", advance="no") "]"
   end subroutine write_vector

end program dump_reference
