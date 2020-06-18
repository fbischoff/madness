/*
  This file is part of MADNESS.

  Copyright (C) 2007,2010 Oak Ridge National Laboratory

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

  For more information please contact:

  Robert J. Harrison
  Oak Ridge National Laboratory
  One Bethel Valley Road
  P.O. Box 2008, MS-6367

  email: harrisonrj@ornl.gov
  tel:   865-241-3937
  fax:   865-572-0680

  $Id$
*/

//#define WORLD_INSTANTIATE_STATIC_TEMPLATES


#include <chem/correlationfactor.h>

namespace madness{

	/// create and return a new nuclear correlation factor

	/// @param[in]	world	the world
	/// @param[in]	calc	the calculation as read from the input file
	/// @return 	a nuclear correlation factor
	std::shared_ptr<NuclearCorrelationFactor>
	create_nuclear_correlation_factor(World& world,
			const Molecule& molecule,
			const std::shared_ptr<PotentialManager> potentialmanager,
			const std::pair<std::string,std::list<double> >& ncf) {

		std::string corrfac=ncf.first;
		std::list<double> factors=ncf.second;
		double a=0.0, b=0.0, c=0.0, d=0.0, e=0.0;
		auto iter=factors.begin();
		if (factors.size()>0) a=*(iter);
		if (factors.size()>1) b=*(++iter);
		if (factors.size()>2) c=*(++iter);
		if (factors.size()>3) d=*(++iter);
		if (factors.size()>4) e=*(++iter);

		int nparam=factors.size()-1;
		Tensor<double> bb(nparam);
		int i=0;
		for (auto iter1 = ++factors.begin(); iter1 != factors.end(); iter1++) {
			bb[i++]=*iter1;
		}


		typedef std::shared_ptr<NuclearCorrelationFactor> ncf_ptr;

		if (corrfac == "gaussslater") {
			return ncf_ptr(new GaussSlater(world, molecule));
		} else if (corrfac == "linearslater") {
			return ncf_ptr(new LinearSlater(world, molecule, a));
        } else if ((corrfac == "gradientalgaussslater") or (corrfac == "ggs")) {
            return ncf_ptr(new GradientalGaussSlater(world, molecule,a));
        } else if (corrfac == "slater") {
			return ncf_ptr(new Slater(world, molecule, a));
        } else if (corrfac == "slaterapprox") {
			return ncf_ptr(new SlaterApprox(world, molecule, a, bb));
        } else if (corrfac == "slaterapprox_h") {
			return ncf_ptr(new SlaterApprox_h(world, molecule, a, b, c));
        } else if (corrfac == "poly4erfc") {
            return ncf_ptr(new poly4erfc(world, molecule, a));
		} else if (corrfac == "polynomial4") {
			return ncf_ptr(new Polynomial<4>(world, molecule, a ));
		} else if (corrfac == "polynomial5") {
			return ncf_ptr(new Polynomial<5>(world, molecule, a));
		} else if (corrfac == "polynomial6") {
			return ncf_ptr(new Polynomial<6>(world, molecule, a));
		} else if (corrfac == "polynomial7") {
			return ncf_ptr(new Polynomial<7>(world, molecule, a));
		} else if (corrfac == "polynomial8") {
			return ncf_ptr(new Polynomial<8>(world, molecule, a));
		} else if (corrfac == "polynomial9") {
			return ncf_ptr(new Polynomial<9>(world, molecule, a));
		} else if (corrfac == "polynomial10") {
			return ncf_ptr(new Polynomial<10>(world, molecule, a));
		} else if ((corrfac == "none") or (corrfac == "one")) {
			return ncf_ptr(new PseudoNuclearCorrelationFactor(world,
					molecule,potentialmanager,1.0));
		} else if (corrfac == "two") {
			return ncf_ptr(new PseudoNuclearCorrelationFactor(world,
					molecule,potentialmanager,2.0));
		} else if (corrfac == "linear") {
			return ncf_ptr(new PseudoNuclearCorrelationFactor(world,
					molecule,potentialmanager, a));
		} else {
			if (world.rank()==0) print(ncf);
			MADNESS_EXCEPTION("unknown nuclear correlation factor", 1);
			return ncf_ptr();
		}
	}

//	std::shared_ptr<NuclearCorrelationFactor>
//	create_nuclear_correlation_factor(World& world,
//			const Molecule& molecule,
//			const std::shared_ptr<PotentialManager> pm,
//			const std::pair<std::string,std::list<double> >& ncf) {
//		std::stringstream ss;
//		ss << ncf.first;
//		for (auto d : ncf.second) ss << " " << d;
//		return create_nuclear_correlation_factor(world,molecule,pm,ss.str());
//	}



	std::shared_ptr<NuclearCorrelationFactor> optimize_approximate_ncf(
			std::shared_ptr<NuclearCorrelationFactor>& ncf_approx,
			const std::shared_ptr<NuclearCorrelationFactor>& ncf,
			std::shared_ptr<real_convolution_3d> poisson,
			const real_function_3d& nemodensity,
			const std::vector<Function<double,3> >& nemo,
			const int nelectron, const double thresh)  {


//		if (SlaterApprox* aaa=dynamic_cast<SlaterApprox*>(ncf_approx.get())) {
//			;
//		} else
//			MADNESS_EXCEPTION("failed to downcast approximate NCF",1);
//		}

		SlaterApprox& slater_approx = dynamic_cast<SlaterApprox&>(*ncf_approx.get());
		double a=slater_approx.get_a();
		Tensor<double> b=copy(slater_approx.get_b());


		// number of parameter listed minus the fixed Slater length parameter a
		int nparam = b.dim(0);

		static double lambda = 1.0;
		print("a = ", a);
		print("b = ", b);
		print("lambda = ", lambda);

		real_function_3d nemodensity_square = nemodensity*nemodensity;
		const real_function_3d R_square=ncf->square();

		//coulomb potential
//		double vj= (((nemodensity*R_square)/(r-r[i])).trace())*nemodensity;;
		real_function_3d vj=(*poisson)(nemodensity*R_square);
		real_function_3d intermediate_j=vj*nemodensity;


		//exchange term

		//exchange potential
		const long nmo=nemo.size();
		std::vector<real_function_3d> vx(nmo*nmo);

		/*for (int i=0; i<nmo; ++i){
			for (int j=0; j<nmo; ++j){
				vx[i*nmo+j] =(*poisson)(nemo[i]*nemo[j]*R_square);
				vx[i*nmo+j] = vx[j*nmo+i];
			}
		}*/

		//difference of approximate and exact density matrices
		std::vector<real_function_3d> delta_rho_ij_div_R_Ra(nmo*nmo);

		//Ex^2 exchange Energie

		real_function_3d intermediate_x=(*poisson)(nemo[0]*nemo[0]*R_square)*nemo[0]*nemo[0]-(*poisson)(nemo[0]*nemo[0]*R_square)*nemo[0]*nemo[0];
				//(*poisson)(nemo[0]*nemo[0]*R_square)*nemo[0]*nemo[0];

		//delta_rho_ij = nemo[i]*nemo[j]*(R_square-R_square_qpprox)
		for (int i=0; i<nmo; ++i){
			for (int j=0; j<nmo; ++j){
				vx[i*nmo+j] =(*poisson)(nemo[i]*nemo[j]*R_square);
				//vx[i*nmo+j] = vx[j*nmo+i];

				delta_rho_ij_div_R_Ra[i*nmo+j] = nemo[i]*nemo[j];
				//delta_rho_ij_div_R_Ra[i*nmo+j] = delta_rho_ij_div_R_Ra[j*nmo+i];

				std::vector<real_function_3d> matrix_elements (nmo*nmo);
				matrix_elements[i*nmo+j] =  delta_rho_ij_div_R_Ra[i*nmo+j]*vx[i*nmo+j];

				intermediate_x = intermediate_x + matrix_elements[i*nmo+j];
			}
		}

		printf("DEBUG\n");

		for(int i=0; i<50; i++){

			slater_approx.set_b(b);
			const real_function_3d R_square_approx=ncf_approx->square();

			save(R_square_approx,"R_square_approx");

			std::vector<real_function_3d> d2Rdb2_div_R_approx(nparam*nparam);	// d^2R / dbdb	-- symmetric matrix
			auto ij = [&nparam](const int i, const int j) {return i*nparam+j;};

			for (int i=0; i<nparam; ++i) {
				for (int j=i; j<nparam; ++j) {
					d2Rdb2_div_R_approx[ij(i,j)]=ncf_approx->d2Rdbdc_div_R2(i,j);
					d2Rdb2_div_R_approx[ij(j,i)]=d2Rdb2_div_R_approx[ij(i,j)];
				}
			}

			std::vector<real_function_3d> dRdb_div_R_approx(nparam);
			for (int i=0; i<nparam; ++i) {
				dRdb_div_R_approx[i]=ncf_approx->dRdb_div_R2(i);
			}

			real_function_3d R_square_approx_times_R_square_approx_minus_R_square = R_square_approx*(R_square_approx-R_square);

			double f=(nemodensity*R_square_approx).trace()-nelectron;
			double g=((nemodensity*R_square-nemodensity*R_square_approx)*(nemodensity*R_square-nemodensity*R_square_approx)).trace();
			double ej=((intermediate_j*R_square-intermediate_j*R_square_approx)*(intermediate_j*R_square-intermediate_j*R_square_approx)).trace();

			double ex = ((intermediate_x*R_square-intermediate_x*R_square_approx)*(intermediate_x*R_square-intermediate_x*R_square_approx)).trace();

			//first derivative of f and g
			Tensor<double> fprime(nparam), gprime(nparam), ejprime(nparam), exprime(nparam);
			for (int i=0; i<nparam; ++i) {
				fprime[i]= 2.0*(nemodensity*R_square_approx*dRdb_div_R_approx[i]).trace();
				gprime[i] = 4.0*(nemodensity_square*R_square_approx_times_R_square_approx_minus_R_square*dRdb_div_R_approx[i]).trace();
				ejprime[i]= 4.0*(intermediate_j*intermediate_j*R_square_approx_times_R_square_approx_minus_R_square*dRdb_div_R_approx[i]).trace();
				exprime[i]= 4.0*(intermediate_x*intermediate_x*R_square_approx_times_R_square_approx_minus_R_square*dRdb_div_R_approx[i]).trace();
			}

			//second derivative of f and g
			Tensor<double> fpp(nparam,nparam), gpp(nparam,nparam), ejpp(nparam,nparam), expp(nparam,nparam);
			for (int i=0; i<nparam; ++i) {
				for (int j=i; j<nparam; ++j) {
					fpp(i,j)=2.0*(nemodensity*(2*R_square_approx*dRdb_div_R_approx[i]*dRdb_div_R_approx[j]+R_square_approx*d2Rdb2_div_R_approx[ij(i,j)])).trace();
					fpp(j,i)=fpp(i,j);

					gpp(i,j)=4.0*(nemodensity_square*(2*dRdb_div_R_approx[i]*dRdb_div_R_approx[j]*R_square_approx*(2*R_square_approx-R_square)
							+R_square_approx*(R_square_approx-R_square)*d2Rdb2_div_R_approx[ij(i,j)])).trace();
					gpp(j,i)=gpp(i,j);

					ejpp(i,j)=4.0*(intermediate_j*intermediate_j*(2*dRdb_div_R_approx[i]*dRdb_div_R_approx[j]*R_square_approx*(2*R_square_approx-R_square)
							+R_square_approx*(R_square_approx-R_square)*d2Rdb2_div_R_approx[ij(i,j)])).trace();
					ejpp(j,i)=ejpp(i,j);

					expp(i,j)=4.0*(intermediate_x*intermediate_x*(2*dRdb_div_R_approx[i]*dRdb_div_R_approx[j]*R_square_approx*(2*R_square_approx-R_square)
							+R_square_approx*(R_square_approx-R_square)*d2Rdb2_div_R_approx[ij(i,j)])).trace();
					expp(j,i)=expp(i,j);

				}
			}

			// gradient of the lagrangian
			Tensor<double> Lprime(nparam+1);
			Lprime(Slice(0,nparam-1))=exprime + lambda*fprime;
			Lprime[nparam]=f;

			// hessian of the lagrangian
			Tensor<double> Lpp(nparam+1,nparam+1);
			Lpp(Slice(0,nparam-1),Slice(0,nparam-1))=expp + lambda*fpp;
			Lpp(nparam,Slice(0,nparam-1))=fprime;								// last line
			Lpp(Slice(0,nparam-1),nparam)=fprime;								// last column
			Lpp(nparam,nparam)=0.0;

			// damp the Hessian
			for (int i=0; i<Lpp.dim(0); ++i) Lpp(i,i)-=0.001;
			Tensor<double> update=-inner(inverse(Lpp),Lprime);
			b=b+update(Slice(0,nparam-1));
			lambda=lambda+update(nparam);

			print("f(b,c) = ", f);
			print("g(b,c) = ", g);
			print ("ej(b,c)= ", ej);
			print ("ex(b,c)= ", ex);
			print("b = ", b);
			print("lambda = ", lambda);

			if (Lprime.normf()<thresh) break;
		}
		return ncf_approx;
	}


}
