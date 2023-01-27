// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2023 by the deal.II authors
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


#include <deal.II/base/logstream.h>

#include <deal.II/lac/ginkgo_sparse_matrix.h>

#ifdef DEAL_II_WITH_GINKGO

#  include <deal.II/lac/exceptions.h>

#  include <ginkgo/core/base/mtx_io.hpp>

#  include <cmath>
#  include <memory>


DEAL_II_NAMESPACE_OPEN

namespace GinkgoWrappers
{

  struct noop_deleter
  {
    template <typename T>
    void
    operator()(T *ptr)
    {}
  };



  template <typename Number, typename IndexType>
  Csr<Number, IndexType>::Csr(std::shared_ptr<const gko::Executor> exec)
    : data_(GkoCsr::create(exec))
  {}

  template <typename Number, typename IndexType>
  Csr<Number, IndexType>::Csr(std::unique_ptr<GkoCsr> M)
    : data_(std::move(M))
  {}

  template class Csr<double, gko::int32>;
  template class Csr<double, gko::int64>;
  template class Csr<float, gko::int32>;
  template class Csr<float, gko::int64>;
  template class Csr<std::complex<double>, gko::int32>;
  template class Csr<std::complex<double>, gko::int64>;
  template class Csr<std::complex<float>, gko::int32>;
  template class Csr<std::complex<float>, gko::int64>;

} // namespace GinkgoWrappers

DEAL_II_NAMESPACE_CLOSE

#endif