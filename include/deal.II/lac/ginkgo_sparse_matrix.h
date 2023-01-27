#ifndef dealii_ginkgo_sparse_matrix_h
#define dealii_ginkgo_sparse_matrix_h


#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_GINKGO

#  include <deal.II/base/index_set.h>
#  include <deal.II/base/subscriptor.h>

#  include <deal.II/lac/ginkgo_vector.h>

#  include <ginkgo/core/matrix/csr.hpp>

#  include <iomanip>
#  include <ios>
#  include <memory>

DEAL_II_NAMESPACE_OPEN

namespace GinkgoWrappers
{

  template <typename Number, typename IndexType = gko::int32>
  class Csr : public Subscriptor
  {
    using GkoCsr = gko::matrix::Csr<Number, IndexType>;

  public:
    using value_type     = Number;
    using real_type      = typename numbers::NumberTraits<Number>::real_type;
    using iterator       = value_type *;
    using const_iterator = const value_type *;
    using size_type      = types::global_dof_index;

    Csr() = delete;

    Csr(std::shared_ptr<const gko::Executor> exec);

    Csr(std::unique_ptr<GkoCsr> M);

    template <typename OtherNumber>
    void
    vmult(Vector<OtherNumber> &u, const Vector<OtherNumber> &v) const;

    template <typename OtherNumber>
    void
    vmult_add(Vector<OtherNumber> &u, const Vector<OtherNumber> &v) const;

    /**
     * @note Since there is no native Ginkgo support for this, this creates a temporary transpose.
     */
    template <typename OtherNumber>
    void
    Tvmult(Vector<OtherNumber> &u, const Vector<OtherNumber> &v) const;

    /**
     * @note Since there is no native Ginkgo support for this, this creates a temporary transpose.
     */
    template <typename OtherNumber>
    void
    Tvmult_add(Vector<OtherNumber> &u, const Vector<OtherNumber> &v) const;

    const GkoCsr *
    get_gko_object() const
    {
      return data_.get();
    }

  private:
    std::unique_ptr<GkoCsr> data_;
  };

  template <typename Number, typename IndexType>
  template <typename OtherNumber>
  void
  Csr<Number, IndexType>::Tvmult_add(Vector<OtherNumber>       &u,
                                     const Vector<OtherNumber> &v) const
  {
    auto one =
      gko::initialize<gko::matrix::Dense<Number>>({1.0}, data_->get_executor());
    data_->transpose()->apply(one.get(),
                              v.get_gko_object().get(),
                              one.get(),
                              u.get_gko_object().get());
  }

  template <typename Number, typename IndexType>
  template <typename OtherNumber>
  void
  Csr<Number, IndexType>::vmult_add(Vector<OtherNumber>       &u,
                                    const Vector<OtherNumber> &v) const
  {
    auto one =
      gko::initialize<gko::matrix::Dense<Number>>({1.0}, data_->get_executor());
    data_->apply(one.get(),
                 v.get_gko_object().get(),
                 one.get(),
                 u.get_gko_object().get());
  }

  template <typename Number, typename IndexType>
  template <typename OtherNumber>
  void
  Csr<Number, IndexType>::Tvmult(Vector<OtherNumber>       &u,
                                 const Vector<OtherNumber> &v) const
  {
    data_->transpose()->apply(v.get_gko_object().get(),
                              u.get_gko_object().get());
  }

  template <typename Number, typename IndexType>
  template <typename OtherNumber>
  void
  Csr<Number, IndexType>::vmult(Vector<OtherNumber>       &u,
                                const Vector<OtherNumber> &v) const
  {
    data_->apply(v.get_gko_object().get(), u.get_gko_object().get());
  }

} // namespace GinkgoWrappers

DEAL_II_NAMESPACE_CLOSE

#endif

#endif // dealii_ginkgo_sparse_matrix_h
