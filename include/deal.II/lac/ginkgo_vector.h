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
    using GkoNormVec = typename gko::matrix::Dense<Number>::absolute_type;

  public:
    using value_type     = Number;
    using real_type      = typename numbers::NumberTraits<Number>::real_type;
    using iterator       = value_type *;
    using const_iterator = const value_type *;
    using size_type      = types::global_dof_index;

    Vector() = delete;

    Vector(std::shared_ptr<const gko::Executor> exec);

    explicit Vector(const size_type                      size,
                    std::shared_ptr<const gko::Executor> exec);
    explicit Vector(const std::initializer_list<Number> &list,
                    std::shared_ptr<const gko::Executor> exec);

    Vector(const Vector &other);
    Vector(Vector &&other) noexcept(false);

    Vector &
    operator=(const Vector &other);
    Vector &
    operator=(Vector &&other) noexcept(false);

    void
    reinit(const Vector &V, const bool omit_zeroing_entries = false);

    iterator
    begin() noexcept;
    const_iterator
    begin() const noexcept;

    iterator
    end() noexcept;
    const_iterator
    end() const noexcept;

    Number
    operator()(const size_type i) const noexcept;
    Number &
    operator()(const size_type i) noexcept;
    Number
    operator[](const size_type i) const noexcept;
    Number &
    operator[](const size_type i) noexcept;

    size_type
    size() const noexcept;

    IndexSet
    locally_owned_elements() const;

    const gko::matrix::Dense<Number> *
    get_gko_object() const noexcept;

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
  typename Vector<Number>::iterator
  Vector<Number>::begin() noexcept
  {
    return data_->get_values();
  }

  template <typename Number>
  typename Vector<Number>::const_iterator
  Vector<Number>::begin() const noexcept
  {
    return data_->get_const_values();
  }


  template <typename Number>
  typename Vector<Number>::iterator
  Vector<Number>::end() noexcept
  {
    return data_->get_values() + size();
  }

  template <typename Number>
  typename Vector<Number>::const_iterator
  Vector<Number>::end() const noexcept
  {
    return data_->get_const_values() + size();
  }

  template <typename Number>
  Number
  Vector<Number>::operator()(const size_type i) const noexcept
  {
    return data_->at(i);
  }

  template <typename Number>
  Number &
  Vector<Number>::operator()(const size_type i) noexcept
  {
    return data_->at(i);
  }

  template <typename Number>
  Number
  Vector<Number>::operator[](const size_type i) const noexcept
  {
    return data_->at(i);
  }

  template <typename Number>
  Number &
  Vector<Number>::operator[](const size_type i) noexcept
  {
    return data_->at(i);
  }

  template <typename Number>
  typename Vector<Number>::size_type
  Vector<Number>::size() const noexcept
  {
    return data_->get_size()[0];
  }

} // namespace GinkgoWrappers

DEAL_II_NAMESPACE_CLOSE

#endif

#endif // dealii_ginkgo_vector_h
