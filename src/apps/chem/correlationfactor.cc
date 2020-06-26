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
			World& world,
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

		const real_function_3d nemodensity_square = nemodensity*nemodensity;
		const real_function_3d R_square=ncf->square();

		//coulomb term

		//coulomb potential
		const real_function_3d vj=(*poisson)(nemodensity*R_square);

		const real_function_3d intermediate_j=vj*nemodensity;


/*
		//exchange term

		const long nmo=nemo.size();

		real_function_3d intermediate_x = real_function_3d(world);

		for (int i=0; i<nmo; ++i){
			for (int j=i; j<nmo; ++j){

				//exchange potential
				real_function_3d vx =(*poisson)(nemo[i]*nemo[j]*R_square);

				
				//product of orbital i and j
				real_function_3d delta_rho_ij_div_R_Ra = nemo[i]*nemo[j];

				
				//matix elements of XC matrix
				real_function_3d matrix_elements =  delta_rho_ij_div_R_Ra*vx; //symmetric matrix

				
				//summation of all matrix elements
				intermediate_x += (i==j) ?  matrix_elements : 2.0*matrix_elements;
			}
		}


*/
		for(int i=0; i<100; i++){

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

			//constraint
			double f=(nemodensity*R_square_approx).trace()-nelectron;

			//choose the error measure subjected to f
			int error_measure = 2;

			/*the different error measures are subjected to the constraint f
			 error measure =1:  g - difference in densities,
			 error measure =2:  e_j - coulomb term,
			 error measure =3:  e_x - exchange term
			*/

			real_function_3d intermediate;
			if(error_measure==1){
				   intermediate = nemodensity;
			}
			else if(error_measure==2){
				intermediate = intermediate_j;
			}
//			else if(error_measure==3){
//					intermediate = intermediate_x;
//			}
			else {
				 print( "error measure is not assigned in 'correlationfactor.cc <226>'");
			}


			real_function_3d intermediate_square = intermediate*intermediate;

			//function subjected to the constraint f
			double function = ((intermediate*R_square-intermediate*R_square_approx)*(intermediate*R_square-intermediate*R_square_approx)).trace();


			//error measure 1-3	
			double g=((nemodensity*R_square-nemodensity*R_square_approx)*(nemodensity*R_square-nemodensity*R_square_approx)).trace();
			double ej=((intermediate_j*R_square-intermediate_j*R_square_approx)*(intermediate_j*R_square-intermediate_j*R_square_approx)).trace();
			//double ex = ((intermediate_x*R_square-intermediate_x*R_square_approx)*(intermediate_x*R_square-intermediate_x*R_square_approx)).trace();

			//first derivative of f and the function subjected to f
			Tensor<double> fprime(nparam), functionprime(nparam);
			for (int i=0; i<nparam; ++i) {
				fprime[i]= 2.0*(nemodensity*R_square_approx*dRdb_div_R_approx[i]).trace();

				functionprime[i]= 4.0*(intermediate_square*R_square_approx_times_R_square_approx_minus_R_square*dRdb_div_R_approx[i]).trace();
			}

			//second derivative of f and the function subjected to f
			Tensor<double> fpp(nparam,nparam), functionpp(nparam,nparam);
			for (int i=0; i<nparam; ++i) {
				for (int j=i; j<nparam; ++j) {
					fpp(i,j)=2.0*(nemodensity*(2*R_square_approx*dRdb_div_R_approx[i]*dRdb_div_R_approx[j]+R_square_approx*d2Rdb2_div_R_approx[ij(i,j)])).trace();
					fpp(j,i)=fpp(i,j);


					functionpp(i,j)=4.0*(intermediate_square*(2*dRdb_div_R_approx[i]*dRdb_div_R_approx[j]*R_square_approx*(2*R_square_approx-R_square)
							+R_square_approx*(R_square_approx-R_square)*d2Rdb2_div_R_approx[ij(i,j)])).trace();
					functionpp(j,i)=functionpp(i,j);
				}
			}

			// gradient of the lagrangian
			Tensor<double> Lprime(nparam+1);
			Lprime(Slice(0,nparam-1))=functionprime + lambda*fprime;
			Lprime[nparam]=f;

			// hessian of the lagrangian
			Tensor<double> Lpp(nparam+1,nparam+1);
			Lpp(Slice(0,nparam-1),Slice(0,nparam-1))=functionpp + lambda*fpp;
			Lpp(nparam,Slice(0,nparam-1))=fprime;								// last line
			Lpp(Slice(0,nparam-1),nparam)=fprime;								// last column
			Lpp(nparam,nparam)=0.0;

			// damp the Hessian
			for (int i=0; i<Lpp.dim(0); ++i) Lpp(i,i)-=0.001;
			Tensor<double> update=-inner(inverse(Lpp),Lprime);
			b=b+update(Slice(0,nparam-1));
			lambda=lambda+update(nparam);

			print("f(b,c) = ", f);
			print("optimized with error measure ", error_measure);
			print("1.  g(b,c)   = ", g);
			print("2. ej(b,c)^2 = ", ej);
//			print("3. ex(b,c)^2 = ", ex);
			print("b = ", b);
			print("lambda = ", lambda);

			if (Lprime.normf()<thresh) break;
		}
		return ncf_approx;
	}


}
