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

#include <deal.II/lac/ginkgo_vector.h>

#ifdef DEAL_II_WITH_GINKGO

#  include <deal.II/lac/exceptions.h>

#  include <ginkgo/core/base/mtx_io.hpp>

#  include <cmath>
#  include <memory>


DEAL_II_NAMESPACE_OPEN

namespace GinkgoWrappers
{
  template <typename Number>
  Vector<Number>::Vector(std::shared_ptr<const gko::Executor> exec)
    : data_(GkoVec::create(std::move(exec)))
  {}

  template <typename Number>
  Vector<Number>::Vector(const size_type                      size,
                         std::shared_ptr<const gko::Executor> exec)
    : data_(GkoVec::create(std::move(exec), gko::dim<2>{size, 1}))
  {}

  template <typename Number>
  Vector<Number>::Vector(const std::initializer_list<Number> &list,
                         std::shared_ptr<const gko::Executor> exec)
    : data_(gko::initialize<GkoVec>(list, std::move(exec)))
  {}
  template <typename Number>
  Vector<Number>::Vector(const Vector<Number> &other)
    : Subscriptor(), data_(gko::clone(other.data_))
  {}

  template <typename Number>
  Vector<Number>::Vector(Vector<Number> &&other) noexcept(false)
    : Vector(other.data_->get_executor())
  {
    data_->move_from(other.data_.get());
  }

  template <typename Number>
  Vector<Number> &
  Vector<Number>::operator=(const Vector<Number> &other)
  {
    if (this != &other)
      {
        data_->copy_from(other.data_.get());
      }
    return *this;
  }

  template <typename Number>
  Vector<Number> &
  Vector<Number>::operator=(Vector<Number> &&other) noexcept(false)
  {
    if (this != &other)
      {
        data_->move_from(other.data_.get());
      }
    return *this;
  }



  template <typename Number>
  const gko::matrix::Dense<Number> *
  Vector<Number>::get_gko_object() const noexcept
  {
    return data_.get();
  }

  template <typename Number>
  std::size_t
  Vector<Number>::memory_consumption() const
  {
    return sizeof(Vector<Number>) + sizeof(GkoVec) + sizeof(Number) * size();
  }

  template <typename Number>
  void
  Vector<Number>::print(std::ostream      &out,
                        const unsigned int precision,
                        const bool         scientific,
                        const bool         accross)
  {
    // TODO: figure out the meaning of accross
    const auto default_precision = out.precision();

    out << std::setprecision(precision);
    if (scientific)
      out << std::scientific;

    gko::write(out, data_.get());

    out << std::setprecision(default_precision);
    if (scientific)
      out << std::defaultfloat;
  }

  template <typename Number>
  IndexSet
  Vector<Number>::locally_owned_elements() const
  {
    return IndexSet(size());
  }

  template <typename Number>
  Number
  Vector<Number>::add_and_dot(const Number a, const Vector &V, const Vector &W)
  {
    this->add(a, V);
    return *this * W;
  }

  template <typename Number>
  typename Vector<Number>::real_type
  Vector<Number>::linfty_norm() const
  {
    throw ExcNotImplemented();
  }

  template <typename Number>
  typename Vector<Number>::real_type
  Vector<Number>::l2_norm() const
  {
    auto result = GkoNormVec::create(data_->get_executor()->get_master(),
                                     gko::dim<2>{1, 1});
    data_->compute_norm2(result.get());
    return result->at(0);
  }

  template <typename Number>
  typename Vector<Number>::real_type
  Vector<Number>::l1_norm() const
  {
    auto result = GkoNormVec ::create(data_->get_executor()->get_master(),
                                      gko::dim<2>{1, 1});
    data_->compute_norm1(result.get());
    return result->at(0);
  }

  template <typename Number>
  Number
  Vector<Number>::mean_value() const
  {
    throw ExcNotImplemented();
  }

  template <typename Number>
  bool
  Vector<Number>::all_zero() const
  {
    auto norm = l1_norm();
    return norm <=
           1e2 * std::numeric_limits<gko::remove_complex<Number>>::min();
  }

  template <typename Number>
  void
  Vector<Number>::reinit(const Vector &V, const bool omit_zeroing_entries)
  {
    data_ = GkoVec::create(data_->get_executor(), V.data_->get_size());
    if (!omit_zeroing_entries)
      {
        data_->fill(0.0);
      }
  }

  template <typename Number>
  Vector<Number> &
  Vector<Number>::operator=(const Number s)
  {
    data_->fill(s);
    return *this;
  }

  template <typename Number>
  void
  Vector<Number>::equ(const Number a, const Vector &V)
  {
    AssertDimension(V.size(), size());
    *this = V;
    *this *= a;
  }

  template <typename Number>
  void
  Vector<Number>::scale(const Vector &scaling_factors)
  {
    AssertDimension(scaling_factors.size(), size());
    auto exec     = data_->get_executor();
    auto col_view = GkoVec::create(exec,
                                   gko::dim<2>{1, size()},
                                   gko::make_array_view(exec, size(), begin()),
                                   size());
    col_view->scale(scaling_factors.data_.get());
  }

  template <typename Number>
  void
  Vector<Number>::sadd(const Number s, const Number a, const Vector &V)
  {
    AssertDimension(V.size(), size());
    *this *= s;
    this->add(a, V);
  }

  template <typename Number>
  void
  Vector<Number>::add(const Number  a,
                      const Vector &V,
                      const Number  b,
                      const Vector &W)
  {
    AssertDimension(V.size(), size());
    AssertDimension(W.size(), size());
    Vector tmp(V);
    tmp.add(b / a, W);
    this->add(a, tmp);
  }

  template <typename Number>
  void
  Vector<Number>::add(const Number a, const Vector &V)
  {
    Assert(V.size() == size(), ExcDimensionMismatch(V.size(), size()));
    auto a_dense = gko::initialize<GkoVec>({a}, data_->get_executor());
    data_->add_scaled(a_dense.get(), V.data_.get());
  }

  template <typename Number>
  void
  Vector<Number>::add(const Number a)
  {
    auto a_vec = Vector(size(), data_->get_executor());
    a_vec.data_->fill(a);
    *this += a_vec;
  }

  template <typename Number>
  Vector<Number> &
  Vector<Number>::operator*=(const Number factor)
  {
    auto factor_dense =
      gko::initialize<GkoVec>({factor}, data_->get_executor());
    data_->scale(factor_dense.get());
    return *this;
  }

  template <typename Number>
  Vector<Number> &
  Vector<Number>::operator/=(const Number factor)
  {
    auto factor_dense =
      gko::initialize<GkoVec>({factor}, data_->get_executor());
    data_->inv_scale(factor_dense.get());
    return *this;
  }

  template <typename Number>
  Vector<Number> &
  Vector<Number>::operator+=(const Vector &V)
  {
    AssertDimension(V.size(), size());
    auto one = gko::initialize<GkoVec>({1.0}, data_->get_executor());
    data_->add_scaled(one.get(), V.data_.get());
    return *this;
  }

  template <typename Number>
  Vector<Number> &
  Vector<Number>::operator-=(const Vector &V)
  {
    AssertDimension(V.size(), size());
    auto neg_one = gko::initialize<GkoVec>({-1.0}, data_->get_executor());
    data_->add_scaled(neg_one.get(), V.data_.get());
    return *this;
  }

  template <typename Number>
  Number
  Vector<Number>::operator*(const Vector &V) const
  {
    AssertDimension(V.size(), size());
    auto result =
      GkoVec ::create(data_->get_executor()->get_master(), gko::dim<2>{1, 1});
    data_->compute_conj_dot(V.data_.get(), result.get());
    return result->at(0);
  }

  template class Vector<double>;
  template class Vector<float>;
  template class Vector<std::complex<double>>;
  template class Vector<std::complex<float>>;

} // namespace GinkgoWrappers


DEAL_II_NAMESPACE_CLOSE

#endif