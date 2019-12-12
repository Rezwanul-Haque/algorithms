# What is divide and conquer?
The divide and conquer paradigm involves breaking a problem into smaller simple sub-problems, and then solving these
sub-problems, and finally, combining the results to obtain a global op‐ timal solution.

# List of divide and conquer algorithms
1. Binary Search
2. Merge sort
3. quick sort
4. Karatsuba algorithm for fast multiplication
5. Strassen's matrix multiplication
6. Closest pair of points

# Long Multiplication probelm defination


## Motivation
Multiplying two four-digit numbers together requires 16 multiplication operations, and we can
generalize and say that an n digit number requires, approximately, $n^2$ multiplication operations


n, is very large this topic is called asymptotic analysis or time complexity.

# Karatsuba Algorithm
Our four-digit number, 2345, becomes a pair of two-digit numbers, 23 and 45. We can
write a more general decomposition of any two n digit numbers, x, and y using the 
following, where m is any positive integer less than n:

x = $10^ma + b$

y = $10^mc + d$

So now we can rewrite our multiplication problem x, y as follows:

($10^ma + b$) ($10^mc + d$)