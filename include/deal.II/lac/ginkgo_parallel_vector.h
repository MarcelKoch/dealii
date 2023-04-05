#ifndef dealii_ginkgo_local_vector_h
#define dealii_ginkgo_local_vector_h


#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_GINKGO

#  include <deal.II/base/index_set.h>
#  include <deal.II/base/partitioner.h>
#  include <deal.II/base/subscriptor.h>

#  include <ginkgo/core/distributed/local_vector.hpp>

#  include <iomanip>
#  include <ios>
#  include <memory>

DEAL_II_NAMESPACE_OPEN

namespace GinkgoWrappers
{

  namespace distributed
  {

    namespace detail
    {
      std::pair<std::vector<int>, gko::array<int>>
      decode_import_data(
        const std::vector<std::pair<unsigned int, unsigned int>> &targets,
        const std::vector<std::pair<unsigned int, unsigned int>> &indices);
    }

    template <typename Number>
    class Vector : public Subscriptor
    {
      using GkoVec = gko::experimental::distributed::LocalVector<Number>;
      //      using GkoNormVec = typename
      //      gko::matrix::Dense<Number>::absolute_type;

    public:
      using value_type     = Number;
      using real_type      = typename numbers::NumberTraits<Number>::real_type;
      using iterator       = value_type *;
      using const_iterator = const value_type *;
      using size_type      = types::global_dof_index;

      Vector() = delete;

      Vector(
        const std::shared_ptr<const gko::Executor>               &exec,
        const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner)
      {
        auto send_data =
          detail::decode_import_data(partitioner->import_targets(),
                                     partitioner->import_indices());

        gko::experimental::mpi::communicator comm(
          partitioner->get_mpi_communicator());

        auto sharing_info = gko::experimental::distributed::
          sharing_info<Number, int>::create_from_send_info(
            comm,
            send_data.first,
            send_data.second,
            gko::experimental::distributed::sharing_mode::set);

        data_ = std::make_unique<GkoVec>(exec,
                                         std::move(comm),
                                         std::move(sharing_info));
      }

      template <typename... Args>
      void
      reinit(Args &&...args)
      {
        *this = Vector(std::forward<Args>(args)...);
      }

      void
      compress(VectorOperation::values op)
      {
        if (op == VectorOperation::insert)
          {
            data_->make_consistent(
              gko::experimental::distributed::sharing_mode::set);
          }
        else if (op == VectorOperation::add)
          {
            data_->make_consistent(
              gko::experimental::distributed::sharing_mode::add);
          }
        else
          {
            // not supported
          }
      }

      void
      update_ghost_values() const
      {
        data_->make_consistent();
      }

      void
      zero_out_ghost_values()
      {
        data_->get_shared()->fill(0.0);
      }

      bool has_ghost_elements() const {
        data_->get_shared()->get_size();
      }

    private:
      std::unique_ptr<GkoVec> data_;
    };

  } // namespace distributed

} // namespace GinkgoWrappers

DEAL_II_NAMESPACE_CLOSE

#endif

#endif // dealii_ginkgo_local_vector_h
