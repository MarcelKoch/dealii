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

// Tests the GinkgoWrappers::Vector interface

#include "../tests.h"

#include "test_macros.h"

// all include files you need here
#include <deal.II/lac/ginkgo_vector.h>


auto exec = gko::ReferenceExecutor::create();


TEST(can_create_from_executor)
{
  GinkgoWrappers::Vector<double> v(exec);

  TEST_ASSERT(v.size() == 0);
}


TEST(can_create_from_ginkgo_object)
{
  GinkgoWrappers::Vector<double> v(
    gko::initialize<gko::matrix::Dense<double>>({1, 2, 3}, exec));

  TEST_ASSERT(v[0] == 1);
  TEST_ASSERT(v[1] == 2);
  TEST_ASSERT(v[2] == 3);
}


TEST(can_create_from_size)
{
  GinkgoWrappers::Vector<double> v(7, exec);

  TEST_ASSERT(v.size() == 7);
}


TEST(can_create_from_initializer_list)
{
  GinkgoWrappers::Vector<double> v({1, 5, 3, 9}, exec);

  TEST_ASSERT(v[0] == 1);
  TEST_ASSERT(v[1] == 5);
  TEST_ASSERT(v[2] == 3);
  TEST_ASSERT(v[3] == 9);
}


TEST(can_copy_construct)
{
  GinkgoWrappers::Vector<double> v({1, 5, 3, 9}, exec);

  GinkgoWrappers::Vector<double> v2(v);

  TEST_ASSERT(v2.size() == v.size());
  TEST_ASSERT(v2.get_gko_object() != v.get_gko_object());
  for (std::size_t i = 0; i < v.size(); ++i)
    {
      TEST_ASSERT(v2[i] == v[i]);
    }
}


TEST(can_move_construct)
{
  GinkgoWrappers::Vector<double> v({1, 5, 3, 9}, exec);

  GinkgoWrappers::Vector<double> v2(std::move(v));

  TEST_ASSERT(v2.size() == 4);
  TEST_ASSERT(v.size() == 0);
  TEST_ASSERT(v2.get_gko_object() != v.get_gko_object());
  TEST_ASSERT(v2[0] == 1);
  TEST_ASSERT(v2[1] == 5);
  TEST_ASSERT(v2[2] == 3);
  TEST_ASSERT(v2[3] == 9);
}


TEST(can_access_ginkgo_object)
{
  GinkgoWrappers::Vector<double> v(10, exec);

  auto obj = v.get_gko_object();

  TEST_ASSERT(obj);
}


TEST(wrapper_same_as_ginkgo_object)
{
  GinkgoWrappers::Vector<double> v({1, 5, 3, 9}, exec);

  auto obj = v.get_gko_object();

  TEST_ASSERT(obj->get_size()[0] == v.size());
  TEST_ASSERT(obj->get_size()[1] == 1);
  for (std::size_t i = 0; i < v.size(); ++i)
    {
      TEST_ASSERT(obj->at(i) == v[i]);
    }
}


TEST(can_reinit)
{
  GinkgoWrappers::Vector<double> v(7, exec);
  GinkgoWrappers::Vector<double> w(3, exec);

  v.reinit(w, false);

  TEST_ASSERT(v.size() == w.size());
  for (size_t i = 0; i < v.size(); ++i)
    {
      TEST_ASSERT(v[i] == 0.0);
    }
}


int
main()
{
  // Initialize deallog for test output.
  // This also reroutes deallog output to a file "output".
  initlog();

  can_create_from_executor();
  can_create_from_ginkgo_object();
  can_create_from_size();
  can_create_from_initializer_list();
  can_copy_construct();
  can_move_construct();
  can_access_ginkgo_object();
  wrapper_same_as_ginkgo_object();

  return 0;
}