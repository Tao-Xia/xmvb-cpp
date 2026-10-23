#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <utility>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "mcscf/ci/factors_internal.hpp"

namespace xmvb::vb {

StructureAction::SourceImageAdjoint StructureAction::project_source_adjoint(
    const mcscf::SpinFactorSource& source, const mcscf::SpinFactorSource& right,
    const ExteriorOrbitalTransform& alpha, const ExteriorOrbitalTransform& beta,
    const mcscf::DirectCiSigmaAction& sigma, const Eigen::MatrixXd* output_generator,
    bool project_output, double residual_energy) const {
  // Image and covector slabs coexist, so partition one budget, not two copies.
  constexpr std::size_t image_bytes = mcscf::kFactorWorkspaceBytes/2;
  mcscf::FactorActionWorkspace workspace(source,sigma.channel_kernel().rows(),
      image_bytes,false,project_output && output_generator ? 2 : 0);
  const int columns = workspace.source_tile()*std::max(1,workspace.channel_tile());
  const std::size_t rows = static_cast<std::size_t>(n_unique_alpha_)+n_unique_beta_;
  const int n = sigma.n_orbitals();
  SourceImageAdjoint result{{Eigen::MatrixXd::Zero(project_output ? n_structures_ : 0,1),
                            Eigen::MatrixXd::Zero(project_output ? n_structures_ : 0,1)},Eigen::MatrixXd::Zero(n,n)};
  if (!right.count) return result;
  const std::size_t reductions = static_cast<std::size_t>(effective_openmp_thread_count()+1)*n*n;
  const std::size_t capacity = (mcscf::kFactorWorkspaceBytes-image_bytes)/sizeof(double);
  const std::size_t fixed = reductions+rows*columns;
  const bool mapped = mcscf::detail::use_amplitude_map(right);
  // Accumulating all right covectors traverses generator links only once.
  // The image-side order avoids all-source residency and H replay, but its
  // generator work follows the number of image factors. Count that work too.
  const long double images=static_cast<long double>(source.count)*(sigma.channel_kernel().rows()+3);
  const long double factor_work=4.0L*rows*right.count;
  const long double links=2.0L*sigma.spin_link_count();
  const std::size_t resident_bytes=(4*rows+columns+right.workspace_rows)*right.count+reductions;
  const bool resident=resident_bytes<=capacity &&
      factor_work*images+links*right.count <=
      (mapped ? right.contract_work:factor_work)*images+links*images;
  if (capacity <= fixed) throw std::length_error("source-image adjoint cannot hold its covector panel");
  const int tile = mapped ? 1:std::min<std::size_t>(right.count,
      (capacity-fixed)/(rows+columns+right.workspace_rows));
  if (!resident && (!tile || fixed+mcscf::detail::amplitude_workspace(right,tile,columns)>capacity))
    throw std::length_error("source-image adjoint cannot hold one amplitude panel");
  std::optional<mcscf::SpinAmplitudeWorkspace> amplitude;
  if (!resident) amplitude.emplace(right,tile,columns);
  Eigen::MatrixXd a(n_unique_alpha_,resident ? right.count:0),b(n_unique_beta_,resident ? right.count:0);
  Eigen::MatrixXd moment(resident ? columns:0,resident ? right.count:0);
  Eigen::MatrixXd za=Eigen::MatrixXd::Zero(n_unique_alpha_,resident ? right.count:columns);
  Eigen::MatrixXd zb=Eigen::MatrixXd::Zero(n_unique_beta_,resident ? right.count:columns);
  workspace::Temporary temporary(sizeof(double)*static_cast<std::size_t>(
      a.size()+b.size()+moment.size()+za.size()+zb.size()));
  if (resident) right.load(0,right.count,a,b);
  const auto consume = [&](const Eigen::Ref<const Eigen::MatrixXd>& x,
        const Eigen::Ref<const Eigen::MatrixXd>& y, double scale) {
      const auto add = [&](const Eigen::Ref<const Eigen::MatrixXd>& left,
          const Eigen::Ref<const Eigen::MatrixXd>& spectator, double weight) {
        if (resident) {
          auto gram=moment.topRows(left.cols());
          mcscf::detail::transpose_product(spectator,b,gram); gram*=weight;
          mcscf::detail::add_product(left,gram,za);
          mcscf::detail::transpose_product(left,a,gram); gram*=weight;
          mcscf::detail::add_product(spectator,gram,zb);
          return;
        }
        auto wa=za.leftCols(left.cols()),wb=zb.leftCols(left.cols());
        wa.setZero(); wb.setZero();
        amplitude->add(left,spectator,wa,wb,weight);
        // <u v^T,E Y> uses Y v and Y^T u. H remains on the image
        // side; no commutation of H with an oriented generator is assumed.
        result.generator += sigma.spin_generator_adjoint(false,left,wa);
        result.generator += sigma.spin_generator_adjoint(true,spectator,wb);
      };
      // Only the left H image needs physical parity reconstruction. Generators
      // commute with common spin exchange; no selected-space closure is used.
      add(x,y,scale*(spin_exchange_parity_ ? 0.5 : 1));
      if (spin_exchange_parity_) add(y,x,scale*0.5*spin_exchange_parity_);
  };
  // Generate each H-image once. Only the narrow amplitude products stream;
  // splitting the other source can no longer replay this physical sweep.
  result.images=project_factor_source(source,workspace,alpha,beta,sigma,false,
      output_generator,consume,project_output,residual_energy);
  if (resident) {
    result.generator+=sigma.spin_generator_adjoint(false,za,a);
    result.generator+=sigma.spin_generator_adjoint(true,zb,b);
  }
  result.images.tiles.source_workspace_bound += sizeof(double)*
      (resident ? resident_bytes:fixed+mcscf::detail::amplitude_workspace(right,tile,columns));
  if (profiling::phase_totals().enabled)
    std::fprintf(stderr,"XMVB_SOURCE_IMAGE_ADJOINT,map=%d,right_resident=%d,source_tile=%d,columns=%d,amplitude_columns=%zu,image_sweeps=1\n",
        !resident && mapped,resident,tile,columns,amplitude ? amplitude->columns():right.count);
  return result;
}

} // namespace xmvb::vb
