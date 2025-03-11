import math
import numpy as np


class MonteCarlo:

    def __init__(self, f, a, b, exact, max_iter=1000, seed=None):
        self.f = f
        self.a = a
        self.b = b
        self.exact = exact
        self.max_iter = 1000
        self.approx_value = 0
        self.error_value = 0
        self.seed = seed

    def get_var(self, a, b, m):
        fx = []
        for i in range(0, m):
            x = np.random.uniform(a, b)
            fx.append(self.f(x))
        return np.var(fx)

    def get_ratio(self, a, b, m):
        res = 1.0
        if a < b:
            mid = (b-a)/2.0
            var_a = self.get_var(a, mid, m)
            var_b = self.get_var(mid, b, m)
        return math.sqrt(var_a/var_b)

    def integrate(self, n):
        m = 100
        r = self.get_ratio(self.a, self.b, m)
        n_a = math.ceil(1/(1+r)*n)
        n_b = n-n_a
        mid = (self.b-self.a)/2
        res = 0
        if 0 < n_a:
            res += self._integrate(self.a, mid, n_a)
        if 0 < n_b:
            res += self._integrate(mid, self.b, n_b)
        return res

    def integrate_non_adaptive(self, n):
        return self._integrate(self.a, self.b, n)

    def _integrate(self, a, b, n):
        if self.seed is not None:
            np.random.seed(self.seed)
        h = (b-a)/n
        res = 0
        for i in range(0, n):
            x = np.random.uniform(self.a, self.b)
            res += self.f(x)
        self.approx_value = h*res
        return self.approx_value

    def error(self, n):
        approx = self.integrate(n)
        self.error_value = abs(approx - self.exact)
        return abs(approx - self.exact)

    def find_degree(self, tol):
        n = 1
        while self.error(n) > tol and n < self.max_iter:
            n += 1
        if self.error_value <= tol:
            print('Tolerance achieved')
        else:
            print('Tolerance not achieved')
        return n

    def print_results(self, n):
        print("Degree: {}".format(n))
        print("Approximation: {}".format(self.approx_value))
        print("Error: {}".format(self.error_value))
        if 1e-3 < self.error_value:
            print("误差比公差更大")
        else:
            print("误差比公差更小")


def run_test(f, a, b, exact):
    monte_carlo = MonteCarlo(f, a, b, exact)
    monte_carlo.error(100)
    monte_carlo.print_results(100)
    monte_carlo.error(1000)
    monte_carlo.print_results(1000)
    monte_carlo.error(10000)
    monte_carlo.print_results(10000)



