// ---------------------------------------------------------------------
//
// Copyright (C) 2015 - 2023 by the deal.II Authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Tests the GinkgoWrappers::Csr interface

#include "../tests.h"

#include "test_macros.h"

// all include files you need here
#include <deal.II/lac/ginkgo_sparse_matrix.h>


auto exec = gko::ReferenceExecutor::create();


TEST(can_create_from_executor)
{
  GinkgoWrappers::Csr<double> m(exec);

  TEST_ASSERT(m.get_gko_object());
}


TEST(can_create_from_ginkgo_object)
{
  auto csr =
    gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                              exec);
  auto orig_csr = gko::clone(csr);

  GinkgoWrappers::Csr<double> m(std::move(csr));

  TEST_ASSERT(m.get_gko_object()->get_size() == orig_csr->get_size());
  TEST_ASSERT(m.get_gko_object()->get_num_stored_elements() ==
              orig_csr->get_num_stored_elements());
  for (size_t i = 0; i < orig_csr->get_size()[0]; ++i)
    {
      TEST_ASSERT(orig_csr->get_row_ptrs()[i] ==
                  m.get_gko_object()->get_const_row_ptrs()[i]);
    }
  for (size_t k = 0; k < orig_csr->get_num_stored_elements(); ++k)
    {
      TEST_ASSERT(orig_csr->get_col_idxs()[k] ==
                  m.get_gko_object()->get_const_col_idxs()[k]);
      TEST_ASSERT(orig_csr->get_values()[k] ==
                  m.get_gko_object()->get_const_values()[k]);
    }
}


TEST(can_vmult)
{
  GinkgoWrappers::Vector<double> v({1, 1, 1}, exec);
  GinkgoWrappers::Vector<double> u({1, 1, 1}, exec);
  GinkgoWrappers::Csr<double> m(
    gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                              exec));

  m.vmult(u, v);

  TEST_ASSERT(u[0] == 6);
  TEST_ASSERT(u[1] == 15);
  TEST_ASSERT(u[2] == 21);
}


TEST(can_vmult_add)
{
  GinkgoWrappers::Vector<double> v({1, 1, 1}, exec);
  GinkgoWrappers::Vector<double> u({1, 1, 1}, exec);
  GinkgoWrappers::Csr<double> m(
    gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                              exec));

  m.vmult_add(u, v);

  TEST_ASSERT(u[0] == 7);
  TEST_ASSERT(u[1] == 16);
  TEST_ASSERT(u[2] == 22);
}


TEST(can_tvmult){
  GinkgoWrappers::Vector<double> v({1, 1, 1}, exec);
  GinkgoWrappers::Vector<double> u({1, 1, 1}, exec);
  GinkgoWrappers::Csr<double> m(
    gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                              exec));

  m.Tvmult(u, v);

  TEST_ASSERT(u[0] == 11);
  TEST_ASSERT(u[1] == 14);
  TEST_ASSERT(u[2] == 17);
}


TEST(can_tvmult_add){
  GinkgoWrappers::Vector<double> v({1, 1, 1}, exec);
  GinkgoWrappers::Vector<double> u({1, 1, 1}, exec);
  GinkgoWrappers::Csr<double> m(
    gko::initialize<gko::matrix::Csr<double>>({{1, 2, 3}, {4, 5, 6}, {6, 7, 8}},
                                              exec));

  m.Tvmult_add(u, v);

  TEST_ASSERT(u[0] == 12);
  TEST_ASSERT(u[1] == 15);
  TEST_ASSERT(u[2] == 18);
}


int
main()
{
  // Initialize deallog for test output.
  // This also reroutes deallog output to a file "output".
  initlog();

  can_create_from_executor();
  can_create_from_ginkgo_object();
  can_vmult();
  can_vmult_add();
  can_tvmult();
  can_tvmult_add();

  return 0;
}