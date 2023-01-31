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

#  include "full_matrix.h"

DEAL_II_NAMESPACE_OPEN

namespace GinkgoWrappers
{
  namespace detail
  {
    template <typename Number>
    std::unique_ptr<gko::matrix::Dense<Number>>
    create_mutable_view(Vector<Number> &v)
    {
      auto exec       = v.get_gko_object()->get_executor();
      auto size       = v.get_gko_object()->get_size();
      auto stride     = v.get_gko_object()->get_stride();
      auto n_elements = v.get_gko_object()->get_num_stored_elements();
      return gko::matrix::Dense<Number>::create(
        exec, size, gko::make_array_view(exec, n_elements, v.begin()), stride);
    }
  } // namespace detail


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
    Csr(std::shared_ptr<const gko::Executor> exec,
        const size_type                      m,
        const size_type                      n);
    Csr(std::unique_ptr<GkoCsr> M);

    Csr(const Csr &) = delete;
    Csr(Csr &&m);

    Csr &
    operator=(const Csr &) = delete;
    Csr &
    operator=(Csr &&m);

    size_type
    m() const;
    size_type
    n() const;

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

    void
    set(const size_type i, const size_type j, const Number value);

    /**
     * @note Since there is no native Ginkgo support for this, this results in repeated calls to set(const size_type, const_size_type, const Number).
     */
    void
    set(const std::vector<size_type> &indices,
        const FullMatrix<Number>     &full_matrix,
        const bool                    elide_zero_values = false);
    /**
     * @note Since there is no native Ginkgo support for this, this results in repeated calls to set(const size_type, const_size_type, const Number).
     */
    void
    set(const std::vector<size_type> &row_indices,
        const std::vector<size_type> &col_indices,
        const FullMatrix<Number>     &full_matrix,
        const bool                    elide_zero_values = false);
    /**
     * @note Since there is no native Ginkgo support for this, this results in repeated calls to set(const size_type, const_size_type, const Number).
     */
    void
    set(const size_type               row,
        const std::vector<size_type> &col_indices,
        const std::vector<Number>    &values,
        const bool                    elide_zero_values = false);
    /**
     * @note Since there is no native Ginkgo support for this, this results in repeated calls to set(const size_type, const_size_type, const Number).
     */
    void
    set(const size_type  row,
        const size_type  n_cols,
        const size_type *col_indices,
        const Number    *values,
        const bool       elide_zero_values = false);

    void
    add(const size_type i, const size_type j, const Number value);

    /**
     * @note Since there is no native Ginkgo support for this, this results in repeated calls to add(const size_type, const_size_type, const Number).
     */
    void
    add(const std::vector<size_type> &indices,
        const FullMatrix<Number>     &full_matrix,
        const bool                    elide_zero_values = true);

    /**
     * @note Since there is no native Ginkgo support for this, this results in repeated calls to add(const size_type, const_size_type, const Number).
     */
    void
    add(const std::vector<size_type> &row_indices,
        const std::vector<size_type> &col_indices,
        const FullMatrix<Number>     &full_matrix,
        const bool                    elide_zero_values = true);

    /**
     * @note Since there is no native Ginkgo support for this, this results in repeated calls to add(const size_type, const_size_type, const Number).
     */
    void
    add(const size_type               row,
        const std::vector<size_type> &col_indices,
        const std::vector<Number>    &values,
        const bool                    elide_zero_values = true);

    /**
     * @note Since there is no native Ginkgo support for this, this results in repeated calls to add(const size_type, const_size_type, const Number).
     */
    void
    add(const size_type  row,
        const size_type  n_cols,
        const size_type *col_indices,
        const Number    *values,
        const bool       elide_zero_values      = true,
        const bool       col_indices_are_sorted = false);

    void
    compress();

    const GkoCsr *
    get_gko_object() const;

  private:
    void
    verify_build_state() const;

    std::unique_ptr<GkoCsr> data_;

    gko::matrix_assembly_data<Number, IndexType> assembly_data_;
  };

  template <typename Number, typename IndexType>
  typename Csr<Number, IndexType>::size_type
  Csr<Number, IndexType>::m() const
  {
    return data_->get_size()[0];
  }

  template <typename Number, typename IndexType>
  typename Csr<Number, IndexType>::size_type
  Csr<Number, IndexType>::n() const
  {
    return data_->get_size()[1];
  }

  template <typename Number, typename IndexType>
  template <typename OtherNumber>
  void
  Csr<Number, IndexType>::Tvmult_add(Vector<OtherNumber>       &u,
                                     const Vector<OtherNumber> &v) const
  {
    verify_build_state();
    auto one =
      gko::initialize<gko::matrix::Dense<Number>>({1.0}, data_->get_executor());
    data_->transpose()->apply(one.get(),
                              v.get_gko_object(),
                              one.get(),
                              detail::create_mutable_view(u).get());
  }

  template <typename Number, typename IndexType>
  template <typename OtherNumber>
  void
  Csr<Number, IndexType>::vmult_add(Vector<OtherNumber>       &u,
                                    const Vector<OtherNumber> &v) const
  {
    verify_build_state();
    auto one =
      gko::initialize<gko::matrix::Dense<Number>>({1.0}, data_->get_executor());
    data_->apply(one.get(),
                 v.get_gko_object(),
                 one.get(),
                 detail::create_mutable_view(u).get());
  }

  template <typename Number, typename IndexType>
  template <typename OtherNumber>
  void
  Csr<Number, IndexType>::Tvmult(Vector<OtherNumber>       &u,
                                 const Vector<OtherNumber> &v) const
  {
    verify_build_state();
    data_->transpose()->apply(v.get_gko_object(),
                              detail::create_mutable_view(u).get());
  }

  template <typename Number, typename IndexType>
  template <typename OtherNumber>
  void
  Csr<Number, IndexType>::vmult(Vector<OtherNumber>       &u,
                                const Vector<OtherNumber> &v) const
  {
    verify_build_state();
    data_->apply(v.get_gko_object(), detail::create_mutable_view(u).get());
  }

} // namespace GinkgoWrappers

DEAL_II_NAMESPACE_CLOSE

#endif

#endif // dealii_ginkgo_sparse_matrix_h
