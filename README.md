
# Distributed Point Function

This crate gives the basic functionality of a cryptographic *distributed point function*, a cryptographic construction first proposed by Gilboa and Ishai [[GI14]](https://www.iacr.org/archive/eurocrypt2014/84410245/84410245.pdf) and later improved by Boyle, Gilboa and Ishai [[BGI16]](https://eprint.iacr.org/2018/707).

With this primitive a unit vector $\vec{e}$ of size $N$ (that is, a vector with zeros in all coordinates except for a single coordinate) can be additively secret shared between two parties such that each share can be described using a key of size $\log N$ while merely assuming the existance of one-way-functions.

## Bug Reporting and Feature Request

In case you find a bug or have an extra feature in mind you are interested in, feel free to submit an issue at the issues page.

Notice: This is a very early stage cryptographic library. USE AT YOUR OWN RISK! ZERO WARRENTY GUARANTEED.

## Changelog

### v0.2.0 (27/9/2022)

- Removed Packed DPF.
- Added DPF iterator.
