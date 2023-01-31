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

// Tests the GinkgoWrappers::Csr manipulation (set/add) interface

#include "../tests.h"

#include "test_macros.h"

// all include files you need here
#include <deal.II/lac/ginkgo_sparse_matrix.h>


auto exec = gko::ReferenceExecutor::create();
using size_type = GinkgoWrappers::Csr<double>::size_type;


TEST(can_set_single_value)
{
  GinkgoWrappers::Csr<double> m(exec, 3, 5);

  m.set(2, 4, 4.4);
  m.compress();

  auto obj = m.get_gko_object();
  TEST_ASSERT(obj->get_num_stored_elements() == 1);
  TEST_ASSERT(obj->get_const_col_idxs()[0] == 4);
  TEST_ASSERT(obj->get_const_values()[0] == 4.4);
  TEST_ASSERT(obj->get_const_row_ptrs()[0] == 0);
  TEST_ASSERT(obj->get_const_row_ptrs()[1] == 0);
  TEST_ASSERT(obj->get_const_row_ptrs()[2] == 0);
  TEST_ASSERT(obj->get_const_row_ptrs()[3] == 1);
}

TEST(can_set_full_matrix)
{
  GinkgoWrappers::Csr<double> m(exec, 3, 5);
  FullMatrix<double> k(2, 2);
  k(0, 0) = 1;
  k(0, 1) = 2;
  k(1, 0) = 0;
  k(1, 1) = 4;
  std::vector<size_type> indices{0, 2};

  m.set(indices, k, false);
  m.compress();

  auto obj = m.get_gko_object();
  TEST_ASSERT(obj->get_num_stored_elements() == 4);
  TEST_ASSERT(obj->get_const_col_idxs()[0] == 0);
  TEST_ASSERT(obj->get_const_col_idxs()[1] == 2);
  TEST_ASSERT(obj->get_const_col_idxs()[2] == 0);
  TEST_ASSERT(obj->get_const_col_idxs()[3] == 2);
  TEST_ASSERT(obj->get_const_values()[0] == 1);
  TEST_ASSERT(obj->get_const_values()[1] == 2);
  TEST_ASSERT(obj->get_const_values()[2] == 0);
  TEST_ASSERT(obj->get_const_values()[3] == 4);
  TEST_ASSERT(obj->get_const_row_ptrs()[0] == 0);
  TEST_ASSERT(obj->get_const_row_ptrs()[1] == 2);
  TEST_ASSERT(obj->get_const_row_ptrs()[2] == 2);
  TEST_ASSERT(obj->get_const_row_ptrs()[3] == 4);
}

int
main()
{
  // Initialize deallog for test output.
  // This also reroutes deallog output to a file "output".
  initlog();

  can_set_single_value();
  can_set_full_matrix();

  return 0;
}