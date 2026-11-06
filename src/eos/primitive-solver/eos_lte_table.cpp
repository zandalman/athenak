//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_lte_table.cpp
//! \brief Table loading for the tabulated LTE primitive-solver policy and the code
//! units of the table.

#include "eos_lte_table.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "eos/eos.hpp"
#include "eos/lte_table_core.hpp"
#include "mesh/mesh.hpp"
#include "eos/lte_table_utils.hpp"
#include "units/units.hpp"

namespace lte_eos {

LTECodeUnits LTECodeUnits::FromUnits(const units::Units &punit) {
  const Real c = units::Units::speed_of_light_cgs;
  if (std::fabs(punit.velocity_cgs()/c - 1.0) > 1.0e-10) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
              << "The tabulated LTE EOS on the dynamical-GR path needs the GR code units "
              << "(<coord>/general_rel = true with <units>/bhmass_msun and "
              << "<units>/density_cgs), in which the velocity unit is c." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return Make(punit.density_cgs(), c);
}

}  // namespace lte_eos

namespace Primitive {

lte_eos::LTETableMeta EOSLTETable::ReadTableFromFile(
    const std::string &fname, const lte_eos::LTECodeUnits &units) {
  lte_table_utils::LTETableHost host = lte_table_utils::ReadLTETableHost(fname, __FILE__);
  lte_eos::HostTable packed = lte_table_utils::PackLTETable(host, units);

  // A fresh allocation, never Kokkos::realloc: a policy copied from another one shares
  // the original's view, and realloc to the same size would overwrite the original's
  // table in place.
  m_data = DvceArray1D<Real>("lte table", packed.data.size());
  Kokkos::View<Real *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_data(packed.data.data(), packed.data.size());
  Kokkos::deep_copy(m_data, host_data);

  m_geom = packed.geom;
  m_geom.f = nullptr;

  // The adiabat label (lte_table_core.hpp): one adiabat per node, integrated with the
  // table's own interpolant from that node to the last density row.  The nodes are
  // independent and each integration costs O(nr) steps, so it runs on the device.  The
  // host then checks it and makes its rows non-decreasing in T within the accuracy of
  // the build (FinalizeAdiabatLabel), and the result is uploaded again.
  {
    const lte_eos::TableView v = View();
    auto data = m_data;
    const size_t label_offset = static_cast<size_t>(lte_eos::kAdLabel)*v.nr*v.nt;
    Kokkos::View<int *, DevMemSpace> status("lte_label_status", v.nr*v.nt);
    Kokkos::View<int *, DevMemSpace> substeps("lte_label_substeps", v.nr*v.nt);
    Kokkos::View<Real *, DevMemSpace> change("lte_label_change", v.nr*v.nt);
    Kokkos::View<Real *, DevMemSpace> ln_amp("lte_label_ln_amp", v.nr*v.nt);
    Kokkos::parallel_for("lte_adiabat_label",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, v.nr*v.nt),
    KOKKOS_LAMBDA(const int idx) {
      const int ir = idx/v.nt;
      const int it = idx - ir*v.nt;
      Real label = 0.0, dl = 0.0, la = 0.0;
      int n = 0;
      status(idx) = lte_eos::BuildAdiabatLabelAtNode(v, ir, it, label, n, dl, &la);
      substeps(idx) = n;
      change(idx) = dl;
      ln_amp(idx) = la;
      data(label_offset + idx) = label;
    });
    Kokkos::deep_copy(host_data, m_data);
    auto h_status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), status);
    auto h_substeps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), substeps);
    auto h_change = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), change);
    auto h_ln_amp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ln_amp);
    lte_table_utils::FinalizeAdiabatLabel(packed, h_status.data(), h_substeps.data(),
                                          h_change.data(), h_ln_amp.data(), units,
                                          __FILE__);
    Kokkos::deep_copy(m_data, host_data);
  }

  // The adiabat label ends at the last density row (lte_eos::LnTFromAdiabatLabel), the
  // largest ln rho whose row coordinate (ln rho - lr0)/d ln rho is at most nr - 1 as the
  // device computes it (a difference and a product, rounded the same way there).
  Real ln_top = m_geom.lr0 + static_cast<Real>(m_geom.nr - 1)/m_geom.idlr;
  while ((ln_top - m_geom.lr0)*m_geom.idlr > static_cast<Real>(m_geom.nr - 1)) {
    ln_top = std::nextafter(ln_top, -std::numeric_limits<Real>::infinity());
  }
  m_ln_n_label_top = ln_top;
  max_n_gas_entropy = std::exp(ln_top);

  min_T = std::exp(m_geom.lt_min);
  m_initialized = true;
  return lte_table_utils::MakeLTETableMeta(host);
}

}  // namespace Primitive
