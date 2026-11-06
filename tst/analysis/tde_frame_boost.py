#!/usr/bin/env python3
"""Compile the production TDE boost kernel in a serial host adapter.

Checks stationary floor cells including AMR roundoff, resolved-fluid boosts,
thermal energy, dual-energy channel selection and conservative restriction.
No GPU or configured build is required; this does not test the evolution solver.
Use --source to check an earlier tde_external.cpp against the same invariants.
"""
from pathlib import Path
import subprocess
import tempfile
root = Path(__file__).resolve().parents[2]
def function(text, start):
    a = text.index(start)
    b = text.index('{', a)
    depth = 1
    c = b + 1
    while depth:
        depth += (text[c] == '{') - (text[c] == '}')
        c += 1
    return text[a:c]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=root/'src/pgen/tde_external.cpp')
src = parser.parse_args().source.read_text()
boost = function(src, 'void BoostHydroState(')
eos = (root/'src/eos/general_c2p_mhd.hpp').read_text()
helpers = '\n'.join(function(eos, x) for x in ['bool MHDEnergyChannelResolvesEint(', 'Real DualEnergyInversionEint('])
call = 'BoostHydroState(&p, x,y,z' + (', h.peos->eos_data.dfloor' if 'const Real rho_boost_min' in boost else '') + ');'
harness = r'''
#include <algorithm>
#include <array>
#include <limits>
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <stdexcept>
using Real=double;
constexpr int IDN=0, IM1=1, IM2=2, IM3=3, IEN=4;
#define SQR(x) ((x)*(x))
#define KOKKOS_LAMBDA [=]
struct View {
 std::shared_ptr<std::vector<Real>> v=std::make_shared<std::vector<Real>>(24,0.);
 Real& operator()(int,int n,int,int,int i)const{return (*v)[i*6+n];}
};
struct EOS {struct {bool use_e=true; Real dfloor=1.;} eos_data;};
struct Hydro {View u0; EOS eos; EOS *peos=&eos; bool use_dual_energy=true; int dual_energy_idx=5; Real dual_energy_eta1=1.e-3;};
struct Mesh {struct {int is=0,ie=3,js=0,je=0,ks=0,ke=0;}mb_indcs;};
struct MeshBlockPack {Hydro *phydro; Mesh *pmesh; int nmb_thispack=1;};
int DevExeSpace(){return 0;}
template<class F>void par_for(const char*,int,int ml,int mu,int kl,int ku,int jl,int ju,int il,int iu,F f){
 for(int m=ml;m<=mu;++m)for(int k=kl;k<=ku;++k)for(int j=jl;j<=ju;++j)for(int i=il;i<=iu;++i)f(m,k,j,i);
}
namespace eos_general {
constexpr Real kDualEnergyMagneticFraction=.1;
HELPERS
}
BOOST
void check(bool ok,const char*msg){if(!ok)throw std::runtime_error(msg);}
bool near(double a,double b){return std::abs(a-b)<1.e-10*std::max(1.,std::max(std::abs(a),std::abs(b)));}
int main(){
 Hydro h; Mesh m; MeshBlockPack p{&h,&m};
 auto run=[&](double x,double y,double z){CALL};
 auto u=h.u0;
 // A stationary floor stays stationary even after AMR rounding it upward.
 const double eps=std::numeric_limits<double>::epsilon();
 const double atmosphere[4]={1., std::nextafter(1.,2.), 1.+8.*eps, 1.+32.*eps};
 for(int i=0;i<4;++i){u(0,0,0,0,i)=atmosphere[i];u(0,4,0,0,i)=.7;u(0,5,0,0,i)=.7;}
 const auto background=*u.v;
 run(10.,20.,-5.);
 check(background==*u.v,"roundoff-level atmosphere received a frame boost");
 std::fill(u.v->begin(),u.v->end(),0.);

 // Two resolved fluid cells have identical velocity and temperature.
 for(int i=0;i<2;++i){double r=2.+i*1.e-8;u(0,0,0,0,i)=r;u(0,1,0,0,i)=.2*r;u(0,2,0,0,i)=-.3*r;u(0,3,0,0,i)=.1*r;u(0,4,0,0,i)=r*(.7+.07);u(0,5,0,0,i)=.1*r;}
 // Third cell is their conservative restriction.
 for(int n=0;n<6;++n)u(0,n,0,0,2)=.5*(u(0,n,0,0,0)+u(0,n,0,0,1));
 // Fourth cell takes the auxiliary channel before and after a large boost.
 u(0,0,0,0,3)=2.;u(0,1,0,0,3)=200.;u(0,4,0,0,3)=10000.01;u(0,5,0,0,3)=.01;
 const auto initial=*u.v;
 run(0.,0.,0.);check(initial==*u.v,"zero boost changed state");
 run(10.,20.,-5.);
 for(int i=0;i<3;++i){
  double r=u(0,0,0,0,i);
  check(near(u(0,1,0,0,i)/r,10.2)&&near(u(0,2,0,0,i)/r,19.7)&&near(u(0,3,0,0,i)/r,-4.9),"nonuniform boost of resolved fluid");
  double k=0.;for(int n=1;n<4;++n)k+=.5*SQR(u(0,n,0,0,i))/r;
  check(near(u(0,4,0,0,i)-k,.7*r),"thermal energy changed");
  check(near(u(0,5,0,0,i),.7*r),"stale auxiliary was not synchronized");
 }
 for(int n=0;n<5;++n)check(near(u(0,n,0,0,2),.5*(u(0,n,0,0,0)+u(0,n,0,0,1))),"boost and restriction do not commute");
 check(near(u(0,5,0,0,3),.01),"auxiliary-channel thermal energy changed");
 h.use_dual_energy=false;
 run(-10.,-20.,5.);
 for(int i=0;i<4;++i)for(int n=0;n<5;++n)check(near(u(0,n,0,0,i),initial[i*6+n]),"inverse boost did not recover conserved state");
 std::cout<<"PASS: stationary rounded atmosphere, fluid thermal preservation, restriction, dual channels, zero and inverse boosts\n";
}
'''.replace('HELPERS',helpers).replace('BOOST',boost).replace('CALL',call)
with tempfile.TemporaryDirectory(prefix='tde_boost_') as td:
    cpp=Path(td)/'test.cpp';cpp.write_text(harness)
    subprocess.run(['c++','-std=c++17','-O2',str(cpp),'-o',td+'/test'],check=True)
    subprocess.run([td+'/test'],check=True)
