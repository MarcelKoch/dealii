#ifndef dealii_ginkgo_vector_h
#define dealii_ginkgo_vector_h


#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_GINKGO

#  include <deal.II/base/subscriptor.h>

#  include <ginkgo/core/matrix/dense.hpp>

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

    const gko::matrix::Dense<Number> *
    get_gko_object() const noexcept
    {
      return data_.get();
    }

  private:
    std::unique_ptr<GkoVec> data_;
  };


} // namespace GinkgoWrappers

DEAL_II_NAMESPACE_CLOSE

#endif

#endif // dealii_ginkgo_vector_h
