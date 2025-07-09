#!/usr/bin/env wolframscript
(* ::Package:: *)

ClearAll[f, x1, x2, x3, p, R, rp, L]


p = 0.5;
R = 1.8;
L = 1.3;
a = (1 - p)/(1 + p);
b = 2/(1 + p);
c = 2 a;
rp = a^a * L^b;
sqrtTerm = Sqrt[
  x3^c + 4 x1^c + 4 (x1 x2)^a
];

numerator = rp R^2 +
   x1^b + x1 x2^a +
   x2 x1^a + x2^b +
   (x3^a + sqrtTerm)/2 * x3;

denominator = 2 * (x1^a + x2^a + (x3^a + sqrtTerm)/2);

f[x1_, x2_, x3_] := numerator / denominator

grad = Grad[f[x1, x2, x3], {x1, x2, x3}];
hess = D[f[x1, x2, x3], {{x1, x2, x3}, 2}];
fNum[x1_, x2_, x3_] := Evaluate[f[x1, x2, x3]]
hessNum = D[fNum[x1, x2, x3], {{x1, x2, x3}, 2}];


hessVal = hessNum /. {x1 -> 0.1, x2 -> 0.1, x3 -> 0.1};
Eigenvalues[hessVal]



