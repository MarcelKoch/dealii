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
  template <typename Number, typename IndexType>
  Csr<Number, IndexType>::Csr(std::shared_ptr<const gko::Executor> exec)
    : data_(GkoCsr::create(exec))
    , assembly_data_({})
  {}

  template <typename Number, typename IndexType>
  Csr<Number, IndexType>::Csr(std::shared_ptr<const gko::Executor> exec,
                              const Csr::size_type                 m,
                              const Csr::size_type                 n)
    : data_(GkoCsr::create(exec, gko::dim<2>{m, n}))
    , assembly_data_(data_->get_size())
  {
  }

  template <typename Number, typename IndexType>
  Csr<Number, IndexType>::Csr(std::unique_ptr<GkoCsr> M)
    : data_(std::move(M))
    , assembly_data_({})
  {}

  template <typename Number, typename IndexType>
  Csr<Number, IndexType> &
  Csr<Number, IndexType>::operator=(Csr &&m)
  {
    if (this != &m)
      {
        data_->move_from(m.data_.get());
        assembly_data_ = std::move(m.assembly_data_);
      }
    return *this;
  }

  template <typename Number, typename IndexType>
  Csr<Number, IndexType>::Csr(Csr &&m)
    : data_(GkoCsr::create(m.get_gko_object()->get_executor()))
    , assembly_data_({})
  {
    *this = std::move(m);
  }

  template <typename Number, typename IndexType>
  const typename Csr<Number, IndexType>::GkoCsr *
  Csr<Number, IndexType>::get_gko_object() const
  {
    return data_.get();
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::set(const Csr::size_type i,
                              const Csr::size_type j,
                              const Number         value)
  {
    assembly_data_.set_value(static_cast<IndexType>(i),
                             static_cast<IndexType>(j),
                             value);
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::set(const Csr::size_type  row,
                              const Csr::size_type  n_cols,
                              const Csr::size_type *col_indices,
                              const Number         *values,
                              const bool            elide_zero_values)
  {
    for (size_type k = 0; k < n_cols; ++k)
      {
        if (values[k] != Number{0} || !elide_zero_values)
          {
            set(row, col_indices[k], values[k]);
          }
      }
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::set(const std::vector<size_type> &indices,
                              const FullMatrix<Number>     &full_matrix,
                              const bool                    elide_zero_values)
  {
    set(indices, indices, full_matrix, elide_zero_values);
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::set(const std::vector<size_type> &row_indices,
                              const std::vector<size_type> &col_indices,
                              const FullMatrix<Number>     &full_matrix,
                              const bool                    elide_zero_values)
  {
    for (size_type row = 0; row < row_indices.size(); ++row)
      {
        set(row_indices[row],
            col_indices.size(),
            col_indices.data(),
            full_matrix[row].begin(),
            elide_zero_values);
      }
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::set(const Csr::size_type          row,
                              const std::vector<size_type> &col_indices,
                              const std::vector<Number>    &values,
                              const bool                    elide_zero_values)
  {
    set(row,
        col_indices.size(),
        col_indices.data(),
        values.data(),
        elide_zero_values);
  }


  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::add(const Csr::size_type i,
                              const Csr::size_type j,
                              const Number         value)
  {
    assembly_data_.add_value(static_cast<IndexType>(i),
                             static_cast<IndexType>(j),
                             value);
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::add(const Csr::size_type  row,
                              const Csr::size_type  n_cols,
                              const Csr::size_type *col_indices,
                              const Number         *values,
                              const bool            elide_zero_values,
                              const bool            col_indices_are_sorted
                              [[maybe_unused]])
  {
    for (size_type k = 0; k < n_cols; ++k)
      {
        if (values[k] != Number{0} || !elide_zero_values)
          {
            add(row, col_indices[k], values[k]);
          }
      }
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::add(const std::vector<size_type> &indices,
                              const FullMatrix<Number>     &full_matrix,
                              const bool                    elide_zero_values)
  {
    add(indices, indices, full_matrix, elide_zero_values);
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::add(const std::vector<size_type> &row_indices,
                              const std::vector<size_type> &col_indices,
                              const FullMatrix<Number>     &full_matrix,
                              const bool                    elide_zero_values)
  {
    for (size_type row = 0; row < row_indices.size(); ++row)
      {
        add(row,
            col_indices.size(),
            col_indices.data(),
            full_matrix[row].begin(),
            elide_zero_values);
      }
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::add(const Csr::size_type          row,
                              const std::vector<size_type> &col_indices,
                              const std::vector<Number>    &values,
                              const bool                    elide_zero_values)
  {
    add(row,
        col_indices.size(),
        col_indices.data(),
        values.data(),
        elide_zero_values);
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::compress()
  {
    AssertThrow(data_->get_num_stored_elements() == 0,
                ExcMessage("compress() can't be called more than once on the "
                           "same object"));
    data_->read(assembly_data_.get_ordered_data());
  }

  template <typename Number, typename IndexType>
  void
  Csr<Number, IndexType>::verify_build_state() const
  {
    AssertThrow(assembly_data_.get_num_stored_elements() == 0,
                ExcInvalidState())
  }

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