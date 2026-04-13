#!/usr/bin/env wolframscript
(* ::Package:: *)

ClearAll[f, x1, x2, x3, p, R, rp]

R = 1.8;
L = 1.3;

f[x1_, x2_, x3_, p_] := Module[
  {a, b, c, r, sqrtTerm, numerator, denominator},	
    a = (1 - p)/(1 + p);
	b = 2/(1 + p);
	c = 2 a;
	r = a^a * L^b;
    sqrtTerm = Sqrt[x3^c + 4 x1^c + 4 (x1 x2)^a];
    numerator = r R^2 + x1^b + x1 x2^a + x2 x1^a + x2^b + (x3^a + sqrtTerm)/2 * x3;
    denominator = 2 * (x1^a + x2^a + (x3^a + sqrtTerm)/2);
    numerator / denominator];


tol = 10^-4;


vals = Range[0.01, 1.0, 0.2]; 
ps = Range[0.2, 0.9, 0.05];


badPoints = Reap[
  Do[
    fnum[x1_, x2_, x3_] := f[x1, x2, x3, p];
    hess = D[fnum[x1, x2, x3], {{x1, x2, x3}, 2}];
    hessVal = Quiet[hess /. {x1 -> xval, x2 -> yval, x3 -> zval}];
    eigs = Quiet[Eigenvalues[hessVal]];
    
    If[ Min[eigs] < tol,
      Sow[{xval, yval, zval, p, Min[eigs]}]
    ],
    {xval, vals}, {yval, vals}, {zval, vals}, {p, ps}
  ]
][[2]];

badPoints

