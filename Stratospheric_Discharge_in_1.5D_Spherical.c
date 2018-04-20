/************************************************************************************
 * REFERENCES:
 * 1) T. Xiong, J.-M. Qiu, Z. Xu, and A. Christlieb // J. Comp. Phys. vol. 273, pp. 618–639 (2014)
 * 2) M. Lesur, Y. Idomura, and X. Garbet // Phys. of Plasmas, vol. 16, no. 9, p. 092305 (2009)
 * 3) Y. Itikawa // J. Phys. Chem. Ref. Data, vol. 35, no. 1, pp. 31–53 (2006)
 * 4) C. Z. Cheng, and G. Knorr // J. Comput. Phys., 22, pp. 330-351 (1976)
 * 5) A. A. Kulikovsky // J. Phys. D: Appl. Phys., 27 (12), pp. 2556-2563 (1994)
 ************************************************************************************/
#include "math.h"
#include "float.h"
#include "stdio.h"
#include "sys/time.h"
#include "stdlib.h"
#include "../COMMON.h"
#ifdef _OPENMP
    #include "omp.h"
#endif

/*************************************************
 * ALL FUNDAMENTAL & TECHNICAL CONSTANTS
 *************************************************/
#define epsilon0_const 8.85418781e-12   /* VACUUM DIELECTRIC PERMITTIVITY [1] */
#define m_const 9.10938356e-31   /* ELECTRON REST MASS [KG] */
#define e_const 1.60217662e-19   /* ELEMENTARY CHARGE [C] */
#define pi_const 3.14159265   /* PI CONSTANT [1] */
#define c_const 2.99792458e+8   /* SPEED OF LIGHT IN VACUUM [M/SEC] */
#define k_const 1.38064852e-23   /* BOLTZMANN CONSTANT [J/K] */

/***************************
 * PROBLEM PARAMETERS
 ***************************/
#define R_c 0.5   /* CLOUD RADIUS [M] */
#define R_i 50.5   /* IONOSPHERE RADIUS [M] */
#define Umax 110.0e+3   /* VOLTAGE [V] */

/**************************************************
 * INITAL NEUTRAL GAS & PLASMA PARAMETERS
 **************************************************/
#define N0 1.0e+9   /* INITIAL PLASMA CONCENTRATION [1/M^3] */
#define W0 (5.0*e_const)   /* THERMAL SPREAD OF THE INITIAL MAXWELLIAN DISTRIBUTION [J] */
#define P0 1.013e+4   /* GAS PRESSURE [PA] */
#define T0 300.0   /* GAS TEMPERATURE [K] */
#define I0 (15.6*e_const)   /* THRESHOLD ENERGY FOR ELECTRON PRODUCTION TERM [J] */

/************************************************************
 * NUMERICAL SCHEME DETAILS & CONTROL PARAMETERS
 ************************************************************/
#define Nr 500   /* SEMI-UNIFORM PHASE-SPACE GRID POINTS */
#define Np 2001
#define grid_r_c 2.0   /* COORDINATE GRID NON-UNIFORMITY PARAMETER */
#define grid_p_c (0.5*(PMAX-PMIN)*Nr/(Nr-1.0)) /* MOMENTUM GRID NON-UNIFORMITY PARAMETER */
#define grid_p_m 0.5   /* MOMENTUM GRID NON-UNIFORMITY PARAMETER */
#define SAVE_TIME_PROFILES 1   /* 1 - TO SAVE CURRENT/VOLTAGE TIME-PROFILES */
#define SAVE_INSTANTS 0   /* 1 - TO SAVE INSTANT FIELD/DENSITIES DISTRIBUTION & EDF */
#define t_save (1.0e-9)   /* TIME INTERVAL FOR SAVING INSTANT FIELD/EDF */

/* SIMULATION END TIME & REGULAR TIME-STEP*/
#define t_step 3.0e-12
#define t_end 100.0e-9

/* COMPUTE CONFORMAL CONSTANTS */
#define Ng (P0/T0/k_const)
#define Pgap (m_const*c_const*sqrt(pow(e_const*Umax/(m_const*c_const*c_const) + 1.0, 2.0) - 1.0))
#define PMIN (-1.0*Pgap)
#define PMAX (1.0*Pgap)

/* GRIDS DISTRIBUTION ([5] p. 73) */
#define grid_r(a,b,c,x) (a + (b - a)*(exp(c*x) - 1.0)/(exp(c) - 1.0))
#define grid_r_prime(a,b,c,x) (c*(b - a)*exp(c*x)/(exp(c) - 1.0))
#define grid_p(c,m,x) (c*x/pow(1.0 - x*x, m))
#define grid_p_prime(c,m,x) (c*(1.0 + (2.0*m - 1.0)*x*x)/pow(1.0 - x*x, m+1.0))
#define grid_xi_inverse(c,p) (p/sqrt(p*p + c*c))

/************************
 * GLOBAL VARIABLES
 ************************/
static double Xi_R[Nr],   /* UNIFORM GRID ARRAY */
                      R[Nr],   /* RADIAL COORDINATE ARRAY */
                      D1R[Nr][Nr],   /* RADIAL COORDINATE FIRST DERIVATIVE MATRIX */
                      Xi_Pr[Np],   /* UNIFORM GRID ARRAY */
                      Pr[Np],   /* RADIAL MOMENTUM ARRAY */
                      EPSILON[Np], GAMMA[Np], VELOCITY[Np],  /* ENERGY, GAMMA-FACTOR & VELOCITY ARRAYS */
                      NU_INELASTIC[Np],   /* FREQUENCY OF INELASTIC COLLISIONS */
                      NU_ELASTIC[Np],   /* FREQUENCY OF ELASTIC COLLISIONS */
                      k_SLOW[Np], p_SLOW[Np],   /* SOURCE COEFFICIENT AND INTERPOLATED MOMENTA */
                      k_FAST[Np], p_FAST[Np];

/****************************
 * FUNCTIONS PROTOTYPES
 ****************************/
static double LAMBERTW1(double x);
static double LINEAR_INTERP(int n, double x[], double data[], double new_x);
static double INELASTIC_CROSS_SECTION(double epsilon);
static double ELASTIC_CROSS_SECTION(double epsilon);
static void POISSON(double ne[Nr], double ni[Nr], double e_r[Nr]);
static void SHIFT_Er(double f[Nr][Np], double dt, double e_r[Nr]);
static void SHIFT_r(double f[Nr][Np], double dt);
static void SHIFT_Pr(double f[Nr][Np], double e_r[Nr], double dt);
static void SHIFT_Q(double f[Nr][Np], double dt);
static void SAVEALL(double f[Nr][Np], char *filename);

/************************************************
 * LAMBERT W-FUNCTION (-1 BRANCH)
 ************************************************/
static double LAMBERTW1(double x)
{
  int I;
  const double eps = 4.0e-16, em1 = 0.3678794411714423215955237701614608;
  double p = 1.0, e, t, w, l1, l2;

  /* CHECK FUNCTION DOMAIN OF REAL-VALUED LAMBERT FUNCTION */
  if ( (x < -em1) || (x > 0.0) || isinf(x) || isnan(x) )   { printf("LambertW: bad argument, exiting!\n"); exit(1); }

  /* RELEASE ASYMPTOTIC PART FOR FURTHER CONSIDERATION */
  if (fabs(x) <= 2.2e-308 )   { return 0.0; }

  /* ITERATIONS PRE-CONDITIONS FOR REST VALUES */
  if ( x < -1.0e-6 )
  {
   p = -sqrt(2.0*(2.7182818284590452353602874713526625*x + 1.0));
   w = -1.0 + p*(1.0 + p*(-0.333333333333333333333 + p*0.152777777777777777777777));
  }
  else   { l1 = log(-x); l2 = log(-l1); w = l1 - l2 + l2/l1; }

  if (fabs(p) < 1.0e-4)   { return w; }

  /* HALLEY ITERATIONS */
  for (I=0; I<100; I++)
  {
   e = exp(w); t = w*e - x; p = w + 1.0; t /= e*p - 0.5*(p + 1.0)*t/p; w -= t;
   if ( fabs(t) < eps*(1.0 + fabs(w)) )   { return w; }
  }

  /* SHOULD NEVER GET HERE! */
  printf("LambertW: inconvergence, exiting!\n");   exit(1);
}

/**********************************************
 * LINEAR INTERPOLATION AT UNIFORM GRID
 **********************************************/
double LINEAR_INTERP(int n, double x[], double data[], double new_x)
{
 const double dx = x[1] - x[0], bias = x[0];
 const int I = (int)((new_x - bias)/dx);
 const double delta = new_x - bias - (double)I*dx;
 const double new_data = data[I] + delta*(data[I+1] - data[I])/dx;

 return new_data;
}

/***************************************************************************
 * EXACT SOLUTION OF POISSON'S EQUATION FOR 1D-SPHERICAL
 * SYMMETRIC CASE PROCEDURE
 ***************************************************************************/
static void POISSON(double ne[Nr], double ni[Nr], double e_r[Nr])
{
 int I, J;
 static double int_r_rho[Nr], int_rho_over_r[Nr];

 /* COMPUTE INDEFINITE INTEGRAL OF q/epsilon0*R*(Ni - Ne) */
 int_r_rho[0] = 0.0;
 for (I=1; I<Nr; I++)
 {
  int_r_rho[I] = int_r_rho[I-1] + e_const/epsilon0_const
                                                 *0.5*(R[I-1]*R[I-1]*(ni[I-1] - ne[I-1]) + R[I]*R[I]*(ni[I] - ne[I]))*(R[I] - R[I-1]);
 }

 /* COMPUTE INDEFINITE INTEGRAL OF 1/R MULTIPLIED BY PREVIOUS ONE */
 int_rho_over_r[0] = 0.0;
 for (I=1; I<Nr; I++)
 {
  int_rho_over_r[I] = int_rho_over_r[I-1]
                             + 0.5*(int_r_rho[I-1]/(R[I-1]*R[I-1]) + int_r_rho[I]/(R[I])*R[I])*(R[I] - R[I-1]);
 }

 /* EXACT EXPRESSIONS FOR THE ELECTRIC FIELD & THE POTENTIAL */
 for (I=0; I<Nr; I++)
 {
  e_r[I] = -(Umax + int_rho_over_r[Nr-1])/(R[I]*R[I]*(1.0/R_c - 1.0/R_i)) + int_r_rho[I]/(R[I]*R[I]);
 }
}

/***********************************************
 * COMPUTE PLASMA COMPONENTS DENSITIES
 ***********************************************/
static void PLASMA(double f[Nr][Np], double e_r[Nr], double ne[Nr], double ni[Nr])
{
 int I, II, J;

 /* FIND ELECTRON DENSITY BY INTEGRATING EDF OVER MOMENTA */
 #pragma omp parallel for default(shared) private(J,II)
 for (I=0; I<Nr; I++)
 {
  ne[I] = 0.0;
  for (J=0; J<Np-1; J++)
  {
   ne[I] += 0.5*(f[I][J] + f[I][J+1])*grid_p_prime(grid_p_c,grid_p_m, 0.5*Xi_Pr[J] + 0.5*Xi_Pr[J+1])
                                                                                                                        *(Xi_Pr[1] - Xi_Pr[0]);
  }

  /* FIND ION DENSITY OUT FROM THE POISSON EQUATION */
  for (II=0; II<Nr; II++)
  {
   ni[I] = ne[I] + epsilon0_const/e_const/(R[I]*R[I])*(D1R[I][II]*R[II]*R[II]*e_r[II]);
  }
 }
}

/****************************
 * SHIFT ELECTRIC FIELD
 ****************************/
static void SHIFT_Er(double f[Nr][Np], double dt, double e_r[Nr])
{
 int I, J;
 double sum, j0 = 0.0;
 static double je[Nr];

 /* COMPUTE ELECTRON CURRENT DENSITY [5] */
 #pragma omp parallel for default(shared) private(J,sum)
 for (I=0; I<Nr; I++)
 {
  sum = 0.0;
  for (J=0; J<Np-1; J++)
  {
   sum += e_const*0.5*(f[I][J]*VELOCITY[J] + f[I][J+1]*VELOCITY[J+1])*
                 grid_p_prime(grid_p_c,grid_p_m, 0.5*Xi_Pr[J] + 0.5*Xi_Pr[J+1])*(Xi_Pr[1] - Xi_Pr[0]);
  }
  je[I] = sum;
 }

 /* COMPUTE FULL CURRENT  */
 for (I=0; I<Nr-1; I++)   { j0 += 0.5*(je[I] + je[I+1])*(R[I+1] - R[I])/(R_i - R_c); }

 /* SOLVE CURRENT BALANCE EQUATION */
 for (I=0; I<Nr; I++)   { e_r[I] += dt*(je[I] - j0)/epsilon0_const; }
}

/************************
 * SHIFT ALONG R-AXIS
 ************************/
static void SHIFT_r(double f[Nr][Np], double dt)
{
 int I, J;
 double SHIFT, u[Nr];

 #pragma omp parallel for default(shared) private(I,u,SHIFT)
 for (J=1; J<Np-1; J++)
 {
  /* SAVE TEMPORARY DATA */
  for (I=0; I<Nr; I++)   { u[I] = f[I][J]; }

  /* PERFORM LINEAR INTERPOLATION */
  for (I=1; I<Nr; I++)
  {
   SHIFT = Xi_R[I] - c_const*Pr[J]/sqrt(m_const*m_const*c_const*c_const + Pr[J]*Pr[J])*dt/
                                                                                grid_r_prime(R_c, R_i, grid_r_c, Xi_R[I]);
   if (SHIFT <= Xi_R[0])   { f[I][J] = u[0]; }
   else if (SHIFT >= Xi_R[Nr-1])   { f[I][J] = u[Nr-1]; }
   else   { f[I][J] = LINEAR_INTERP(Nr, Xi_R, u, SHIFT); }
  }

  /* NEUMANN BCS */
  f[0][J] = f[1][J];
 }
}

/***********************
 * SHIFT ALONG P-AXIS
 ***********************/
static void SHIFT_Pr(double f[Nr][Np], double e_r[Nr], double dt)
{
 int I, J;
 double SHIFT, u[Np];

 #pragma omp parallel for default(shared) private(J,u,SHIFT)
 for (I=1; I<Nr; I++)
 {
  /* SAVE TEMPORARY DATA */
  for (J=0; J<Np; J++)   { u[J] = f[I][J]; }

  /* PERFORM LINEAR INTERPOLATION */
  for (J=1; J<Np-1; J++)
  {
   SHIFT = Xi_Pr[J] + e_const*e_r[I]*dt/grid_p_prime(grid_p_c, grid_p_m, Xi_Pr[J]);
   if ( (SHIFT <= Xi_Pr[0]) || (SHIFT >= Xi_Pr[Np-1]) )   { f[I][J] = 0.0; }
   else   { f[I][J] = LINEAR_INTERP(Np, Xi_Pr, u, SHIFT); }
  }
 }
}

/*******************************************
 * ACCOUNTING ALL COLLISION TERMS
 *******************************************/
static void SHIFT_Q(double f[Nr][Np], double dt)
{
 int I, J;
 double f_temp[Np], QMINUS[Np];

 #pragma omp parallel for default(shared) private(J,QMINUS,f_temp)
 for (I=0; I<Nr; I++)
 {
  for (J=0; J<Np; J++)
  {
   QMINUS[J] = NU_INELASTIC[J]*f[I][J];
   f_temp[J] = f[I][J];
  }
  for (J=1; J<Np-1; J++)
  {
   f[I][J] += dt*(k_FAST[J]*LINEAR_INTERP(Np, Xi_Pr, QMINUS, grid_xi_inverse(grid_p_c, p_FAST[J]))
                  + k_SLOW[J]*LINEAR_INTERP(Np, Xi_Pr, QMINUS, grid_xi_inverse(grid_p_c, p_SLOW[J]))
                                         - NU_ELASTIC[J]*(f_temp[J] - LINEAR_INTERP(Np, Xi_Pr, f_temp, -Xi_Pr[J]))
                                                                                                                        - QMINUS[J])/GAMMA[J];
  }
 }
}

/*********************************************************
 * INELASTIC (IONIZATION) CROSS-SECTION IN NITROGEN
 *********************************************************/
double INELASTIC_CROSS_SECTION(double epsilon)
{
 int J;
 double SIGMA, EV = epsilon/e_const;
 /* NELDER APPROXIMATION COEFFICIENT PROVIDING DATA FROM TABLE 15-17 [3]*/
 const double a = -16.85027, b0 = 10.32416, b1 = 0.18589, b2 = 0.00111;
 const double z = 6.4, i = 10.6*e_const;

 /* CROSS-SECTION DATA NELDER FUNCTIONAL FORM APPROXIMATION FROM [3] */
 if (EV <= 1000.0)   { SIGMA = (EV + a)/(b0 + b1*(EV + a) + b2*pow(EV + a, 2.0))*1.0e-20; }
 else   { SIGMA = 2.0/pi_const*z*pow(e_const*e_const/(4.0*epsilon0_const)/epsilon, 2.0)*(epsilon/i - 1.0); }

 /* PRESERVE CROSS-SECTION POSITIVE SIGN */
 if (SIGMA < 0)   { SIGMA = 0.0; }

 return (SIGMA);
}

/******************************************************************
 * ELASTIC (MOMENTUM-TRANSFER) CROSS-SECTION IN NITROGEN
 ******************************************************************/
double ELASTIC_CROSS_SECTION(double epsilon)
{
 int J;
 double SIGMA, EV = epsilon/e_const;
 const double a = 0.01768, b0 = 0.01084, b1 = 0.07527, b2 = 0.00255;
 const double z = 8.2, i = 50.26*e_const;

 /* CROSS-SECTION DATA NELDER FUNCTIONAL FORM APPROXIMATION FROM [3] */
 if (EV <= 100.0)   { SIGMA = (EV + a)/(b0 + b1*(EV + a) + b2*pow(EV + a, 2.0))*1.0e-20; }
 /* HIGH-ENERGY BETHE LIMIT */
 else
 {
  SIGMA = 2.0*pi_const/4.0*pow(z*e_const*e_const/(4.0*pi_const*epsilon0_const)/epsilon, 2.0)*log((epsilon/i)*(epsilon/i));
 }

 return (SIGMA);
}

/******************************
 * SAVE EDF INTO NAMED FILE
 ******************************/
static void SAVEALL(double f[Nr][Np], char *filename)
{
 int I, J;
 FILE *file;

 file = fopen(filename, "wt");
 for (I=0; I<Nr; I++)
 {
  for (J=0; J<Np; J++)
  {
   fprintf(file, "%.6e\t", f[I][J]);
  }
  fprintf(file, "\n");
 }
 fclose(file);
}

int main()
{
 /* LOCAL ONE- & TWO-DIMENSIONAL ARRAYS */
 static double f[Nr][Np], Er[Nr], Ne[Nr], Ni[Nr];

 /* GENERAL PURPOSE VARIABLES */
 int I, II, J, JJ, T = 0, t1, t2;
 double W12, Jfull, duration, time = 0.0, delta1, delta2;
 static char fn[512];
 int t_number = -1;
 FILE *filename1;

 /* (R, Pr) - PHASE-SPACE GRIDS */
 for (I=0; I<Nr; I++)   { Xi_R[I] = I/(Nr - 1.0); }
 for (I=0; I<Nr; I++)   { R[I] = grid_r(R_c, R_i, grid_r_c, Xi_R[I]); }
 for (J=0; J<Np; J++)   { Xi_Pr[J] = -1.0 + 2.0*J/(Np - 1.0); }
 for (J=0; J<Np; J++)   { Pr[J] = grid_p(grid_p_c, grid_p_m, Xi_Pr[J]); }

 /* SET INFINITE MOMENTUM TO FINITE MAX VALUES */
 Pr[0] = -1.0e+123; Pr[Np-1] = 1.0e+123;

 /* DERIVATIVE MATRIX (2ND ORDER ACCURACY) ON NON-UNIFORM GRID */
 for (I=1; I<Nr-1; I++)
 {
  delta1 = R[I] - R[I-1];   delta2 = R[I+1] - R[I];
  D1R[I][I-1] = -delta2/delta1/(delta1 + delta2);
  D1R[I][I] = (delta2 - delta1)/delta1/delta2;
  D1R[I][I+1] = delta1/delta2/(delta1 + delta2);
 }
 delta1 = R[1] - R[0];   delta2 = R[2] - R[1];
 D1R[0][0] = -(2.0*delta1 + delta2)/delta1/(delta1 + delta2);
 D1R[0][1] = (delta1 + delta2)/(delta1*delta2);
 D1R[0][2] = -delta1/delta2/(delta1 + delta2);

 delta1 = R[Nr-1] - R[Nr-2];   delta2 = R[Nr-2] - R[Nr-3];
 D1R[Nr-1][Nr-3] = delta1/delta2/(delta1 + delta2);
 D1R[Nr-1][Nr-2] = -(delta1 + delta2)/delta1/delta2;
 D1R[Nr-1][Nr-1] = (2.0*delta1 + delta2)/delta1/(delta1 + delta2);

 /* SAVE PHASE-SPACE MESH */
 filename1 = fopen("R.dat", "wt");
 for (I=0; I<Nr; I++)   { fprintf(filename1, "%.6e\n", R[I]); }
 fclose(filename1);
 filename1 = fopen("Pr.dat", "wt");
 for (J=0; J<Np; J++)   { fprintf(filename1, "%.6e\n", Pr[J]); }
 fclose(filename1);

 /* COMPUTE ENERGY & GAMMA-FACTOR */
 for (J=0; J<Np; J++)
 {
  GAMMA[J] = sqrt(1.0 + pow(Pr[J]/(m_const*c_const), 2.0));
  EPSILON[J] = m_const*c_const*c_const*(GAMMA[J] - 1.0);
  VELOCITY[J] = Pr[J]/(m_const*GAMMA[J]);
 }
 VELOCITY[0] = -c_const;   VELOCITY[Np-1] = c_const;

 /* SAVE ENERGIES */
 filename1 = fopen("Epsilon.dat", "wt");
 for (J=0; J<Np; J++)
 {
  if (Pr[J] > 0.0)   { fprintf(filename1, "%.6e\n", EPSILON[J]/e_const); }
  else   { fprintf(filename1, "%.6e\n", -EPSILON[J]/e_const); }
 }
 fclose(filename1);

 /* PRE-CALCULATE BOLTZMANN EQUATION R.H.S. PARTS */
 for (J=1; J<Np-1; J++)
 {
  /* COLLISION FREQUENCY FOR ELASTIC & INELASTIC COLLISIONS */
  NU_ELASTIC[J] = Ng*ELASTIC_CROSS_SECTION(EPSILON[J])*fabs(VELOCITY[J]);
  NU_INELASTIC[J] = Ng*INELASTIC_CROSS_SECTION(EPSILON[J])*fabs(VELOCITY[J]);
 }

 /* COMPUTE MOMENTA & COLLISION INTEGRAL COEFFICIENTS */
 for (J=0; J<Np; J++)
 {
  /* SLOW */
  W12 = I0*exp(EPSILON[J]/I0);
  if (W12 < EPSILON[Np-1])
  {
   if ( Pr[J] > 0.0 )
   {
    p_SLOW[J] = m_const*c_const*sqrt(pow(1.0 + W12/(m_const*c_const*c_const), 2.0) - 1.0);
    k_SLOW[J] = Pr[J]/p_SLOW[J]*(1.0 + W12/(m_const*c_const*c_const))/GAMMA[J]*exp(EPSILON[J]/I0);
   }
   else if ( Pr[J] < 0.0 )
   {
    p_SLOW[J] = -m_const*c_const*sqrt(pow(1.0 + W12/(m_const*c_const*c_const), 2.0) - 1.0);
    k_SLOW[J] = Pr[J]/p_SLOW[J]*(1.0 + W12/(m_const*c_const*c_const))/GAMMA[J]*exp(EPSILON[J]/I0);
   }
  }

  /* FAST */
  /* EXACT EXPRESSION WITH LAMBERT-W FUNCTION (-1 BRANCH) */
  W12 = - I0*LAMBERTW1(-exp(-1.0 - EPSILON[J]/I0));
  if (W12 > 0.0)
  {
   if ( Pr[J] > 0.0 )
   {
    p_FAST[J] = m_const*c_const*sqrt(pow(1.0 + W12/(m_const*c_const*c_const), 2.0) - 1.0);
    k_FAST[J] = Pr[J]/p_FAST[J]*(1.0 + W12/(m_const*c_const*c_const))/GAMMA[J]/(1.0 + 1.0/(LAMBERTW1(-exp(-1.0 - EPSILON[J]/I0))));
   }
   else if ( Pr[J] < 0.0 )
   {
    p_FAST[J] = -m_const*c_const*sqrt(pow(1.0 + W12/(m_const*c_const*c_const), 2.0) - 1.0);
    k_FAST[J] = Pr[J]/p_FAST[J]*(1.0 + W12/(m_const*c_const*c_const))/GAMMA[J]/(1.0 + 1.0/(LAMBERTW1(-exp(-1.0 - EPSILON[J]/I0))));
   }
  }
  /* FORCE TO USE W(x) = log(-x) - log(-log(-x)) ASYMPTOTIC EXPRESSION FOR -0 LIMIT */
  else
  {
   W12 =  I0 + EPSILON[J] + I0*log(1.0 + EPSILON[J]/I0);
   if ( Pr[J] > 0.0 )
   {
    p_FAST[J] = m_const*c_const*sqrt(pow(1.0 + W12/(m_const*c_const*c_const), 2.0) - 1.0);
    k_FAST[J] = Pr[J]/p_FAST[J]*(1.0 + W12/(m_const*c_const*c_const))/GAMMA[J];
   }
   else if ( Pr[J] < 0.0 )
   {
    p_FAST[J] = -m_const*c_const*sqrt(pow(1.0 + W12/(m_const*c_const*c_const), 2.0) - 1.0);
    k_FAST[J] = Pr[J]/p_FAST[J]*(1.0 + W12/(m_const*c_const*c_const))/GAMMA[J];
   }
  }
 }

 /* INITIAL MAXWELLIAN EDF & UNIFORM ION DENSITY PROFILE */
 for (I=0; I<Nr; I++)
 {
  for (J=1; J<Np-1; J++)
  {
   f[I][J] = N0/sqrt(2.0*pi_const*m_const*W0)*exp(-pow(Pr[J], 2.0)/(2.0*m_const*W0));
  }
  Ni[I] = N0;   Ne[I] = N0;
 }

 /* PRECOMPUTE INITIAL ELECTRIC FIELD DISTRIBUTION
     IN THE CASE WHERE THE INITIAL ELECTRIC FIELD NON-ZERO */
 POISSON(Ne, Ni, Er);

 /******************************************
  * MAIN NUMERICAL SOLUTION ROUTINE
  ******************************************/
 t1 = rtclock();
 do
 {
  /* SAVE INSTANT DISTRIBUTION OF THE ELECTRIC FIELD, DENSITIES & EDF */
  #if (SAVE_INSTANTS == 1)
  if ( (int)(time/t_save) > t_number )
  {
   /* SAVE EDF */
   sprintf(fn, "EDF_%d.dat", t_number+1);
   SAVEALL(f, fn);
   sprintf(fn, "DISTRIBUTIONS_%d.dat", t_number+1);

   /* COMPUTE IONS & ELECTRONS NUMBER OF PARTICLES */
   PLASMA(f, Er, Ne, Ni);

   /* SAVE ELECTRON & ION DENSITIES, AS WELL AS THE ELECTRIC FIELD */
   filename1 = fopen(fn, "wt");
   for (I=0; I<Nr; I++)   { fprintf(filename1, "%.6e\t %.6e\t %.6e\n", Ne[I], Ni[I], Er[I]); }
   fclose(filename1);
   t_number++;
  }
  #endif

  /* PERFORM ONE TIME STEP */
  SHIFT_r(f, 0.5*t_step);
     SHIFT_Q(f, 0.5*t_step);
        SHIFT_Er(f, 0.5*t_step, Er);
           SHIFT_Pr(f, Er, t_step);
        SHIFT_Er(f, 0.5*t_step, Er);
     SHIFT_Q(f, 0.5*t_step);
  SHIFT_r(f, 0.5*t_step);

  /* PROGRESS INDICATOR */
  printf("\r  Complete %.2f %%", time/t_end*100.0);   fflush(stdout);

  /* SET NEXT TIME STEP */
  time += t_step;
 } while (time<=t_end); t2 = rtclock();   duration = t2 - t1;

 /* DONE! */
 if (duration >= 3600.0)
 {
  printf("\r  Solution procedure complete after %.2f hours!\n", duration/3600.0);
 }
 else if ((duration >= 60.0) && (duration <3600))
 {
  printf("\r  Solution procedure complete after %.0f minutes!\n", duration/60.0);
 }
 else
 {
  printf("\r  Solution procedure complete after %.0f seconds!\n", duration);
 }

 /* SAVE FINAL PLASMA DISTRIBUTION */
 PLASMA(f, Er, Ne, Ni);

 /* SAVE FINAL EDF */
 SAVEALL(f, "EDF.dat");

 /* SAVE FINAL SPACE DISTRIBUTIONS */
 filename1 = fopen("DISTRIBUTIONS.dat", "wt");
 for (I=0; I<Nr; I++)   { fprintf(filename1, "%.6e\t %.6e\t %.6e\n", Ne[I], Ni[I], Er[I]); }
 fclose(filename1);

 return 0;
}
