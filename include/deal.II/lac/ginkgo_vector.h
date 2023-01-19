#ifndef dealii_ginkgo_vector_h
#define dealii_ginkgo_vector_h


#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_GINKGO

#  include <deal.II/base/index_set.h>
#  include <deal.II/base/subscriptor.h>

#  include <ginkgo/core/matrix/dense.hpp>

#  include <iomanip>
#  include <ios>
#  include <memory>

DEAL_II_NAMESPACE_OPEN

namespace GinkgoWrappers
{

  template <typename Number>
  class Vector : public Subscriptor
  {
    using GkoVec = gko::matrix::Dense<Number>;

  public:
    using value_type     = Number;
    using real_type      = typename numbers::NumberTraits<Number>::real_type;
    using iterator       = value_type *;
    using const_iterator = const value_type *;
    using size_type      = types::global_dof_index;

    Vector() = delete;

    Vector(std::shared_ptr<const gko::Executor> exec)
      : data_(GkoVec::create(std::move(exec)))
    {}

    explicit Vector(const size_type                      size,
                    std::shared_ptr<const gko::Executor> exec)
      : data_(GkoVec::create(std::move(exec), gko::dim<2>{size, 1}))
    {}
    explicit Vector(const std::initializer_list<Number> &list,
                    std::shared_ptr<const gko::Executor> exec)
      : data_(gko::initialize<GkoVec>(list, std::move(exec)))
    {}

    Vector(const Vector &other)
      : data_(gko::clone(other.data_))
    {}
    Vector(Vector &&other) noexcept(false)
      : Vector(other.data_->get_executor())
    {
      data_->move_from(other.data_.get());
    };

    Vector &
    operator=(const Vector &other)
    {
      if (this != &other)
        {
          data_->copy_from(other.data_.get());
        }
      return *this;
    }
    Vector &
    operator=(Vector &&other) noexcept(false)
    {
      if (this != &other)
        {
          data_->move_from(other.data_.get());
        }
      return *this;
    }

    void
    reinit(const Vector &V, const bool omit_zeroing_entries = false);


    iterator
    begin() noexcept
    {
      return data_->get_values();
    }
    const_iterator
    begin() const noexcept
    {
      return data_->get_const_values();
    }

    iterator
    end() noexcept
    {
      return data_->get_values() + size();
    }
    const_iterator
    end() const noexcept
    {
      return data_->get_const_values() + size();
    }

    Number
    operator()(const size_type i) const noexcept
    {
      return data_->at(i);
    }
    Number &
    operator()(const size_type i) noexcept
    {
      return data_->at(i);
    }
    Number
    operator[](const size_type i) const noexcept
    {
      return data_->at(i);
    }
    Number &
    operator[](const size_type i) noexcept
    {
      return data_->at(i);
    }

    size_type
    size() const noexcept
    {
      return data_->get_size()[0];
    }

    IndexSet
    locally_owned_elements() const;

    const gko::matrix::Dense<Number> *
    get_gko_object() const noexcept
    {
      return data_.get();
    }

    Vector &
    operator=(const Number s);

    Vector &
    operator*=(const Number factor);

    Vector &
    operator/=(const Number factor);

    Vector &
    operator+=(const Vector &V);
    Vector &
    operator-=(const Vector &V);

    Number
    operator*(const Vector &V) const;

    void
    add(const Number a);

    void
    add(const Number a, const Vector &V);

    /**
     * @note Since there is no native Ginkgo support for this, this uses a copy and two add operations
     */
    void
    add(const Number a, const Vector &V, const Number b, const Vector &W);

    /**
     * @note Since there is no native Ginkgo support for this, this uses a scale and add operation.
     */
    void
    sadd(const Number s, const Number a, const Vector &V);

    void
    scale(const Vector &scaling_factors);

    /**
     * @note Since there is no native Ginkgo support for this, this uses a copy and a scale operation.
     */
    void
    equ(const Number a, const Vector &V);

    /**
     * @note Since there is no native Ginkgo support for this, this computes the L1 norm and compares against 100 * minimal_number
     * @return
     */
    bool
    all_zero() const;

    /**
     * @warning Not implemented
     */
    Number
    mean_value() const;

    real_type
    l1_norm() const;

    real_type
    l2_norm() const;

    /**
     * @warning Not implemented
     */
    real_type
    linfty_norm() const;

    /**
     * @note Since there is no native Ginkgo support for this, this uses an add and a scalar product operation     * @return
     */
    Number
    add_and_dot(const Number a, const Vector &V, const Vector &W);

    void
    print(std::ostream  &out,
          const unsigned precision  = 3,
          const bool     scientific = true,
          const bool     accross    = true);

    std::size_t memory_consumption() const;


  private:
    std::unique_ptr<GkoVec> data_;
  };
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

    out << std::setprecision(default_precision);
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
  real_type
  Vector<Number>::linfty_norm() const
  {
    throw ExcNotImplemented();
  }

  template <typename Number>
  real_type
  Vector<Number>::l2_norm() const
  {
    auto result =
      GkoVec::create(data_->get_executor()->get_master(), gko::dim<2>{1, 1});
    data_->compute_norm2(result.get());
    return result->at(0);
  }

  template <typename Number>
  real_type
  Vector<Number>::l1_norm() const
  {
    auto result =
      GkoVec::create(data_->get_executor()->get_master(), gko::dim<2>{1, 1});
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
    return norm <= 1e2 * std::numeric_limits<gko::remove_complex<Number>>::min()
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
} // namespace GinkgoWrappers

DEAL_II_NAMESPACE_CLOSE

#endif

#endif // dealii_ginkgo_vector_h
